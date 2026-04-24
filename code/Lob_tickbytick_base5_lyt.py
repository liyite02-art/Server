import pandas as pd
import polars as pl
import numpy as np
import glob, gc, os
import pickle
import warnings
import duckdb
import argparse
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from pathlib import Path
from numba import njit
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# 基础配置
con = duckdb.connect()

ORDER_TYPE_MARKET = 1
ORDER_TYPE_LIMIT = 2
ORDER_TYPE_BEST = 3
BS_FLAG_BUY = 1
BS_FLAG_SELL = 2
MATCH_TYPE_UNKNOW = 0
MATCH_TYPE_DEAL = 1
MATCH_TYPE_DEL = 2
INNER_FLAG_OUT = 1
INNER_FLAG_IN = 2

# ============================================================
# 文件路径
# ============================================================
# 请注意以下路径无需修改
# 框架相关路径
global_root_path = rf'/home/lwyxyz'
data79_root_path = rf'{global_root_path}/2.79'
minute_fea_path = rf'{global_root_path}/Stock60sBaseDataAll/Feather'
minute_mmap_path = rf'{global_root_path}/Stock60sBaseDataAll/Mmap'
support_data_path = rf'{global_root_path}/Stock60sConfig/support_data'

# 日频原始数据
data79_root_path = rf'{global_root_path}/2.79'
stock_daily_data_path1 = rf'{data79_root_path}/tonglian_data/ohlc_fea'
stock_daily_data_path2 = rf'{data79_root_path}/tonglian_data/support_data'
stock_daily_data_path3 = rf'{data79_root_path}/update/短周期依赖数据'

# 高频原始数据
local_data_path = rf'{global_root_path}/2.79/ftp/mdl_fea'
trans_data_path = rf'{local_data_path}/trans_fea'
order_data_path = rf'{local_data_path}/order_fea'
lob_data_path = rf'{global_root_path}/hft_database/nas3/sec_lobdata'
lob_data_all_path = rf'{global_root_path}/hft_database/nas3/lobdata_all'

# 逐笔LOB数据路径
tickbytick_lob_path = "/home/lwyxyz/253.118/lob_data"
# 逐笔成交数据路径
data_dir = "/home/lwyxyz/2.79/ftp/mdl_fea"

# ============================================================
# 通用函数
# ============================================================
# 重新定义时间戳，将时间戳转换为整数格式 HHMMSSmmm（毫秒固定为000），通过该函数reindex timestamp列
def generate_minute_timestamps():
    # 定义时间段
    morning_start = pd.Timestamp('09:30:00')
    morning_end = pd.Timestamp('11:29:00')
    afternoon_start = pd.Timestamp('13:00:00')
    afternoon_end = pd.Timestamp('14:56:00')

    # 以分钟为单位生成
    freq = '1min'
    morning_times = pd.date_range(start=morning_start, end=morning_end, freq=freq)
    afternoon_times = pd.date_range(start=afternoon_start, end=afternoon_end, freq=freq)

    # 合并
    all_times = morning_times.append(afternoon_times)

    # 转换为整数格式 HHMMSSmmm（毫秒固定为000）
    time_ints = all_times.strftime('%H%M%S000').astype(np.int64)

    return time_ints

def get_all_trade_days():
    read_file = rf"{stock_daily_data_path2}/trade_days_dict.pkl"
    all_trade_days = pd.read_pickle(read_file)['trade_days']
    all_trade_days = [x.strftime('%Y%m%d') for x in all_trade_days]
    all_trade_days.sort()
    return all_trade_days


def get_trade_days(start_date, end_date):
    all_trade_days = get_all_trade_days()
    trade_days = [date for date in all_trade_days if start_date <= date <= end_date]
    return trade_days


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
    
def read_all_support_dict():
    support_dict_list = [x for x in os.listdir(support_data_path) if "_loc_dict" in x]
    all_dict = {}
    for x in support_dict_list:
        try:
            dict_name = x.split("trade_")[1].split("_loc_dict")[0]
            all_dict[dict_name] = load_pickle(rf"{support_data_path}/{x}")
        except:
            pass
    return all_dict


# 整合时间戳不均匀的LOB数据，通过twap方式整合
def aggregate_lob_with_fill(df, interval='100ms'):
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])

    price_cols = [c for c in df.columns if c.startswith('bp') or c.startswith('sp')] + ['price']
    size_cols  = [c for c in df.columns if c.startswith('bv') or c.startswith('sv')] 
    vol_cols   = ['vol','cnt','amt']
    
    df[price_cols + size_cols] = df[price_cols + size_cols].replace(0, np.nan)

    agg_dict = {col: 'mean' for col in price_cols + size_cols}
    agg_dict.update({col: 'sum' for col in vol_cols})

    results = []
    for code, g in df.groupby('code'):
        g = g.sort_values('time')

        agg_df = (
            g.groupby(pd.Grouper(key='time', freq=interval, offset='9h30min'))
              .agg(agg_dict)
              .reset_index()
        )

        start_day = g['time'].dt.normalize().iloc[0]
        morning_times = pd.date_range(
            start=start_day + pd.Timedelta(hours=9, minutes=30),
            end=start_day + pd.Timedelta(hours=11, minutes=29),
            freq=interval
        )
        afternoon_times = pd.date_range(
            start=start_day + pd.Timedelta(hours=13, minutes=0),
            end=start_day + pd.Timedelta(hours=14, minutes=56),
            freq=interval
        )

        full_times = morning_times.union(afternoon_times)

        full_df = pd.DataFrame({'time': full_times})

        # 合并并前向填充（只在交易时间段填充）
        merged = pd.merge(full_df, agg_df, on='time', how='left')
        merged[price_cols + size_cols] = merged[price_cols + size_cols].ffill()
        merged[vol_cols] = merged[vol_cols].fillna(0)
        merged.insert(0, 'code', code)
        results.append(merged)

    return pd.concat(results, ignore_index=True)


# 整合时间戳不均匀的LOB数据，通过重采样last的方式整合
def align_with_last_snapshot(df, interval='100ms'):
    """
    将数据对齐到固定间隔的网格点，取每个网格点前最后一个有效快照
    参数:
        df: 原始DataFrame
        interval: 对齐间隔，默认为100ms
    返回:
        对齐后的DataFrame
    """
    # 确保时间戳为datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])

    # 创建目标时间网格
    start_time = df['time'].min().ceil(interval)
    end_time = df['time'].max().floor(interval)
    target_times = pd.date_range(start=start_time, end=end_time, freq=interval)

    # ===== 时间筛选逻辑（仅保留交易时段）=====
    # 获取当天日期（假设单日数据）
    day_start = df['time'].dt.normalize().iloc[0]

    morning_start = day_start + pd.Timedelta(hours=9, minutes=30)
    morning_end   = day_start + pd.Timedelta(hours=11, minutes=30)
    afternoon_start = day_start + pd.Timedelta(hours=13, minutes=0)
    afternoon_end   = day_start + pd.Timedelta(hours=14, minutes=57)

    target_times = target_times[
        ((target_times >= morning_start) & (target_times < morning_end)) |
        ((target_times >= afternoon_start) & (target_times < afternoon_end))
    ]

    target_df = pd.DataFrame({'target_time': target_times})

    # 使用merge_asof进行向后对齐（取每个目标时间之前的最新记录）
    aligned_df = pd.merge_asof(
        target_df,
        df,
        left_on='target_time',
        right_on='time',
        direction='backward'
    )

    # 设置目标时间为新的时间戳
    aligned_df['time'] = aligned_df['target_time']
    aligned_df.drop(columns=['target_time'], inplace=True)

    return aligned_df

def align_minute_describe(group):
    df = group.copy()
    df['price'] = df['price'].replace(0, np.nan)
    df['price_fill'] = df['price'].fillna((df['bp1'] + df['sp1']) / 2)

    ohlcv = df.resample('1min', on='time').agg(
        open=('price_fill','first'),
        high=('price_fill','max'),
        low=('price_fill','min'),
        close=('price_fill','last'),
        volume=('vol','sum')
    )

    # 固定 237 分钟索引
    trade_date = group['time'].dt.normalize().iloc[0]
    morning_times = pd.date_range(trade_date + pd.Timedelta('09:30:00'),
                                  trade_date + pd.Timedelta('11:29:00'), freq='1min')
    afternoon_times = pd.date_range(trade_date + pd.Timedelta('13:00:00'),
                                    trade_date + pd.Timedelta('14:56:00'), freq='1min')
    all_times = morning_times.append(afternoon_times)
    ohlcv = ohlcv.reindex(all_times)
    
    ohlcv['close'] = ohlcv['close'].ffill()
    for col in ['open','high','low']:
        ohlcv[col] = ohlcv[col].fillna(ohlcv['close'])
    ohlcv['volume'] = ohlcv['volume'].fillna(0)
    return ohlcv


# 等间隔重采样237个数据点
def sample_237(df):
    n = len(df)
    idx = np.linspace(0, n - 1, 237, dtype=int)
    idx = np.clip(idx, 0, n - 1)  # 防止浮点误差导致越界
    return df.iloc[idx]

# 处理单逻辑单个group的字段计算
def process_group(cfg_dict, group):
    """处理单个分组的所有逻辑"""
    results = {}
    for logic_name, cfg in cfg_dict.items():
        results[logic_name] = cfg['func'](group)
    return results

def zscore_normalize(df):
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    factor_cols = [x for x in df.columns if x not in ["code", "second"]]

    df = df.with_columns([
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
        for c in factor_cols
    ])

    quantiles = (
        df.group_by("code")
        .agg(
            [pl.col(c).quantile(0.01).alias(f"{c}_low") for c in factor_cols] +
            [pl.col(c).quantile(0.99).alias(f"{c}_high") for c in factor_cols]
        )
    )
    df = df.join(quantiles, on="code", how="left")

    df = df.with_columns([
        pl.col(c).clip(pl.col(f"{c}_low"), pl.col(f"{c}_high")).alias(c)
        for c in factor_cols
    ])

    stats = (
        df.group_by("code")
        .agg(
            [pl.col(c).mean().alias(f"{c}_mean") for c in factor_cols] +
            [pl.col(c).std().alias(f"{c}_std") for c in factor_cols]
        )
    )
    df = df.join(stats, on="code", how="left")

    df = df.with_columns([
        pl.when(
                pl.col(f"{c}_std").is_null()
                | pl.col(f"{c}_std").is_nan()
                | (pl.col(f"{c}_std") <= EPS)
            )
          .then(pl.lit(float("nan")))
          .otherwise((pl.col(c) - pl.col(f"{c}_mean")) / pl.col(f"{c}_std"))
          .alias(c + "_z")
        for c in factor_cols
    ])

    return df.select(["code", "second"] + [c + "_z" for c in factor_cols]).to_pandas()

def format_second_data(second_data):
    float_cols = [x for x in second_data.columns if x not in ["code", "second"]]
    # 所有字段按float64保存
    second_data[float_cols] = second_data[float_cols].astype('float64')
    # 这里需要把列名改为code+second+字段名的顺序，否则后续转存为mmap时会有异常
    second_data = second_data[["code", "second"] + float_cols]
    return second_data

def rename_columns(df: pd.DataFrame, logic_name: str) -> pd.DataFrame:
    """默认重命名规则：对除 'code' 和 'second' 以外的列，统一命名为 {logic_name}_{原列名}。
    保持 'code' 和 'second' 原名用于后续按这两个字段合并/对齐。
    """
    rename_map = {col: f"{logic_name}_{col}" for col in df.columns if col not in ("code", "second")}
    return df.rename(columns=rename_map)

def add_tick_fixed_suffix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_rename = {col: f"{col}_tick_fixed" for col in df.columns if col not in ['code', 'second']}
    return df.rename(columns=cols_to_rename)

def aggregate_ms_lob(base_path, specific_date):
    print(f"开始处理日期: {specific_date}")
    with duckdb.connect() as con:
        # 处理深圳市场数据
        trans_sz_df = pd.read_feather(f"{data_dir}/order_trans_sz/{specific_date}.fea")
        trans_sz_df = pl.from_pandas(trans_sz_df)
        
        lob_sz_df = pl.read_parquet(f"{base_path}/lob_data_sz/{specific_date}.parquet")
        
        # 注册DataFrame到DuckDB
        con.register("lob_sz_df", lob_sz_df)
        con.register("trans_sz_df", trans_sz_df)
        
        # 执行LOB数据聚合
        sz_df_aggr = con.sql(f"""
            select code, time,
                arg_max(price, seq_num) as price,
                {', '.join([f'arg_max({col}, seq_num) as {col}' for col in LOB_cols])}
            from lob_sz_df
            group by code, time
            order by code, time
        """).pl()
        
        # 执行交易数据聚合
        sz_df_aggr_2 = con.sql("""
            select code[1:6]::int as code, time, 
                sum(if(type='T', volume, 0))::int as vol,
                sum(if(type='T', 1, 0))::int as cnt,
                sum(if(type='T', volume*price, 0)) as amt
            from trans_sz_df
            group by code, time
            order by code, time
        """).pl()
        
        # 合并LOB和交易数据
        lob_sz = con.sql("""
            select *
            from sz_df_aggr_2
            join sz_df_aggr using(code, time)
            order by code, time
        """).pl()

        # 处理上海市场数据
        trans_sh_df = pd.read_feather(f"{data_dir}/order_trans_sh/{specific_date}.fea")
        trans_sh_df = pl.from_pandas(trans_sh_df)
        
        lob_sh_df = pl.read_parquet(f"{base_path}/lob_data_sh/{specific_date}.parquet")

        # 重新注册上海市场DataFrame
        con.register("lob_sh_df", lob_sh_df)
        con.register("trans_sh_df", trans_sh_df)

        # 执行LOB数据聚合
        sh_df_aggr = con.sql(f"""
            select code, time,
                arg_max(price, seq_num) as price,
                {', '.join([f'arg_max({col}, seq_num) as {col}' for col in LOB_cols])}
            from lob_sh_df
            group by code, time
            order by code, time
        """).pl()
        
        # 执行交易数据聚合
        sh_df_aggr_2 = con.sql("""
            select code[1:6]::int as code, time, 
                sum(if(type='T', volume, 0))::int as vol,
                sum(if(type='T', 1, 0))::int as cnt,
                sum(if(type='T', volume*price, 0)) as amt
            from trans_sh_df
            group by code, time
            order by code, time
        """).pl()
        
        # 合并LOB和交易数据
        lob_sh = con.sql("""
            select *
            from sh_df_aggr_2
            join sh_df_aggr using(code, time)
            order by code, time
        """).pl()
    
    return lob_sh, lob_sz



def load_daily_groups(specific_date):
    base_path = f'{tickbytick_lob_path}'
    lob_sh, lob_sz = aggregate_ms_lob(base_path, specific_date)
    dfs = [lob_sh, lob_sz]
    if any(df is None for df in dfs):
        return None
    print(" - parquet 文件读取完成")
    
    df = pl.concat(dfs, how="vertical")
    print(f" - 合并完成，共 {df.height:,} 行，{df.width} 列")
    
    df = df.with_columns(pl.col("code").cast(pl.Utf8).str.zfill(6))
    
    year = int(specific_date[:4])
    month = int(specific_date[4:6])
    day = int(specific_date[6:8])
    
    ts_int = pl.col("time").cast(pl.Int64)
    df = df.with_columns([
        ((ts_int // 10000000)).alias("hours"),
        ((ts_int // 100000) % 100).alias("minutes"),
        ((ts_int // 1000) % 100).alias("seconds"),
        ((ts_int % 1000) * 1000).alias("microseconds"),
    ])

    df = df.with_columns(
        pl.datetime(
            year, month, day,
            pl.col("hours"), pl.col("minutes"), pl.col("seconds"), pl.col("microseconds")
        ).alias("time")
    ).drop(["hours", "minutes", "seconds", "microseconds"])
    
    start_filter = pl.datetime(year, month, day, 9, 25, 0)
    end_filter   = pl.datetime(year, month, day, 14, 57, 0)
    df = df.filter((pl.col("time") >= start_filter) & (pl.col("time") < end_filter))

    print(" - 按 code 分组")
    grouped = df.partition_by("code", maintain_order=True)

    groups = {}
    for g in grouped:
        code = g["code"][0]
        groups[code] = g.to_pandas()

    return list(groups.items())

def aggregate_ms_lob_sz(base_path,specific_date):
    print(f"开始处理日期: {specific_date}")
    with duckdb.connect() as con:
        # 处理深圳市场数据
        trans_sz_df = pd.read_feather(f"{data_dir}/order_trans_sz/{specific_date}.fea")
        trans_sz_df = pl.from_pandas(trans_sz_df)
        
        lob_sz_df = pl.read_parquet(f"{base_path}/lob_data_sz/{specific_date}.parquet")
        
        # 注册DataFrame到DuckDB
        con.register("lob_sz_df", lob_sz_df)
        con.register("trans_sz_df", trans_sz_df)
        
        # 执行LOB数据聚合
        sz_df_aggr = con.sql(f"""
            select code, time,
                arg_max(price, seq_num) as price,
                {', '.join([f'arg_max({col}, seq_num) as {col}' for col in LOB_cols])}
            from lob_sz_df
            group by code, time
            order by code, time
        """).pl()
        
        # 执行交易数据聚合
        sz_df_aggr_2 = con.sql("""
            select code[1:6]::int as code, time, 
                sum(if(type='T', volume, 0))::int as vol,
                sum(if(type='T', 1, 0))::int as cnt,
                sum(if(type='T', volume*price, 0)) as amt
            from trans_sz_df
            group by code, time
            order by code, time
        """).pl()
        
        # 合并LOB和交易数据
        lob_sz = con.sql("""
            select *
            from sz_df_aggr_2
            join sz_df_aggr using(code, time)
            order by code, time
        """).pl()

    return lob_sz

def load_lob_sh(lob_sz, specific_date):
    sz_codes = set(lob_sz['code'].cast(pl.Utf8).str.zfill(6).unique().to_list())
    print(f'深市股票数量：{len(sz_codes)}')

    folder_path = f"{lob_data_all_path}/{specific_date}"
    file_paths = glob.glob(os.path.join(folder_path, "*.fea"))
    if not file_paths:
        print("没有fea文件")
        return None

    sh_files = []
    for f in file_paths:
        base = os.path.basename(f)
        code = os.path.splitext(base)[0]
        if code not in sz_codes:
            sh_files.append(f)
    print(f"沪市股票数量: {len(sh_files)}")

    df_list = []
    for f in tqdm(sh_files, desc="读取SH fea文件"):
        try:
            df = pd.read_feather(f)[1:]
            df_list.append(df)
        except Exception as e:
            print(f"读取 {f} 出错: {e}")

    if not df_list:
        print("没有可用的 SH 数据")
        return None

    lob_sh = pd.concat(df_list, ignore_index=True)
    mask = (lob_sh['timestamp'] >= 93000000) & (lob_sh['timestamp'] <= 145700000)
    lob_sh = lob_sh.loc[mask]
    bp_cols_sh = [f"bp_{i}" for i in range(1, 6)]
    bv_cols_sh = [f"bv_{i}" for i in range(1, 6)]
    sp_cols_sh = [f"sp_{i}" for i in range(1, 6)]
    sv_cols_sh = [f"sv_{i}" for i in range(1, 6)]
    bn_cols_sh = [f"bn_{i}" for i in range(1, 6)]
    sn_cols_sh = [f"sn_{i}" for i in range(1, 6)]

    base_cols = ["code", "timestamp", "price", "vol", "amt", "num"]
    keep_cols = base_cols + bp_cols_sh + bv_cols_sh + sp_cols_sh + sv_cols_sh + bn_cols_sh + sn_cols_sh
    keep_cols = [c for c in keep_cols if c in lob_sh.columns]
    lob_sh = lob_sh[keep_cols]

    rename_dict = {"timestamp": "time"}
    rename_dict.update({f"bp_{i}": f"bp{i}" for i in range(1, 6)})
    rename_dict.update({f"bv_{i}": f"bv{i}" for i in range(1, 6)})
    rename_dict.update({f"sp_{i}": f"sp{i}" for i in range(1, 6)})
    rename_dict.update({f"sv_{i}": f"sv{i}" for i in range(1, 6)})
    rename_dict.update({f"bn_{i}": f"bn{i}" for i in range(1, 6)})
    rename_dict.update({f"sn_{i}": f"sn{i}" for i in range(1, 6)})
    rename_dict.update({"num": "cnt"})
    
    lob_sh = lob_sh.rename(columns=rename_dict)
    lob_sh = pl.from_pandas(lob_sh)
    
    cols = (
        [f"bp{i}" for i in range(1, 6)] +
        [f"bv{i}" for i in range(1, 6)] +
        [f"sp{i}" for i in range(1, 6)] +
        [f"sv{i}" for i in range(1, 6)] +
        [f"bn{i}" for i in range(1, 6)] +
        [f"sn{i}" for i in range(1, 6)]
    )
    abs_cols = [pl.col(c).abs().alias(c) for c in cols if c in lob_sh.columns]

    lob_sh = lob_sh.with_columns(abs_cols)
    
    casts = [lob_sh[col].cast(lob_sz[col].dtype) for col in lob_sh.columns if col in lob_sz.columns]
    lob_sh = lob_sh.with_columns(casts)
    return lob_sh

def load_daily_groups_before_20210426(specific_date):
    base_path = f'{tickbytick_lob_path}'
    lob_sz = aggregate_ms_lob_sz(base_path, specific_date)
    lob_sh = load_lob_sh(lob_sz, specific_date)
    
    dfs = [lob_sh, lob_sz]
    if any(df is None for df in dfs):
        return None
    print(" - parquet 文件读取完成")
    
    # df = pl.concat(dfs, how="vertical")
    common_cols = [c for c in lob_sh.columns if c in lob_sz]
    df = pl.concat([lob_sh.select(common_cols), lob_sz.select(common_cols)], how="vertical")

    print(f" - 合并完成，共 {df.height:,} 行，{df.width} 列")
    
    df = df.with_columns(pl.col("code").cast(pl.Utf8).str.zfill(6))
    
    year = int(specific_date[:4])
    month = int(specific_date[4:6])
    day = int(specific_date[6:8])
    
    ts_int = pl.col("time").cast(pl.Int64)
    df = df.with_columns([
        ((ts_int // 10000000)).alias("hours"),
        ((ts_int // 100000) % 100).alias("minutes"),
        ((ts_int // 1000) % 100).alias("seconds"),
        ((ts_int % 1000) * 1000).alias("microseconds"),
    ])

    df = df.with_columns(
        pl.datetime(
            year, month, day,
            pl.col("hours"), pl.col("minutes"), pl.col("seconds"), pl.col("microseconds")
        ).alias("time")
    ).drop(["hours", "minutes", "seconds", "microseconds"])
    
    start_filter = pl.datetime(year, month, day, 9, 25, 0)
    end_filter   = pl.datetime(year, month, day, 14, 57, 0)
    df = df.filter((pl.col("time") >= start_filter) & (pl.col("time") <= end_filter))

    print(" - 按 code 分组")
    grouped = df.partition_by("code", maintain_order=True)

    groups = {}
    for g in grouped:
        code = g["code"][0]
        groups[code] = g.to_pandas()

    return list(groups.items())

def load_daily_groups_before_20200922(specific_date):
    folder_path = f'{lob_data_all_path}/{specific_date}'
    file_paths = glob.glob(os.path.join(folder_path, '*.fea'))
    if not file_paths:
        return
    
    groups = {}
    for f in tqdm(file_paths, desc="读取fea文件"):
        try:
            df = pd.read_feather(f)[1:]
            df = df[(df['timestamp']>=93000000) & (df['timestamp']<=145700000)]
            
            ts_int = df['timestamp'].to_numpy(dtype=np.int64)
            hours = ts_int // 10000000
            minutes = (ts_int // 100000) % 100
            seconds = (ts_int // 1000) % 100
            microseconds = (ts_int % 1000) * 1000

            time_delta = (
                pd.to_timedelta(hours, unit='h')
                + pd.to_timedelta(minutes, unit='m')
                + pd.to_timedelta(seconds, unit='s')
                + pd.to_timedelta(microseconds, unit='us')
            )
            day_start = pd.Timestamp(specific_date)
            df['timestamp'] = day_start + time_delta
            
            cols_to_convert = df.columns.difference(['code', 'timestamp'])
            df[cols_to_convert] = df[cols_to_convert].astype(np.float32)

            code = df["code"].iloc[0]
            
            bp_cols_df = [f"bp_{i}" for i in range(1, 6)]
            bv_cols_df = [f"bv_{i}" for i in range(1, 6)]
            sp_cols_df = [f"sp_{i}" for i in range(1, 6)]
            sv_cols_df = [f"sv_{i}" for i in range(1, 6)]
            bn_cols_df = [f"bn_{i}" for i in range(1, 6)]
            sn_cols_df = [f"sn_{i}" for i in range(1, 6)]

            base_cols = ["code", "timestamp", "price", "vol", "amt", "num"]
            keep_cols = base_cols + bp_cols_df + bv_cols_df + sp_cols_df + sv_cols_df + bn_cols_df + sn_cols_df
            keep_cols = [c for c in keep_cols if c in df.columns]
            df = df[keep_cols]

            rename_dict = {"timestamp": "time"}
            rename_dict.update({f"bp_{i}": f"bp{i}" for i in range(1, 6)})
            rename_dict.update({f"bv_{i}": f"bv{i}" for i in range(1, 6)})
            rename_dict.update({f"sp_{i}": f"sp{i}" for i in range(1, 6)})
            rename_dict.update({f"sv_{i}": f"sv{i}" for i in range(1, 6)})
            rename_dict.update({f"bn_{i}": f"bn{i}" for i in range(1, 6)})
            rename_dict.update({f"sn_{i}": f"sn{i}" for i in range(1, 6)})
            rename_dict.update({"num": "cnt"})
            df = df.rename(columns=rename_dict)
            
            if code in groups:
                groups[code] = pd.concat([groups[code], df], ignore_index=True)
            else:
                groups[code] = df

        except Exception as e:
            print(f"读取{f}出错: {e}")
      
    return list(groups.items())

def process_one_day(specific_date, main_logic_config, base_output, n_workers, use_multiprocess):
    if specific_date < "20200922" or specific_date == '20221206' or specific_date == '20201012' or specific_date == '20201014' or specific_date == '20201019' or specific_date == '20201030' or specific_date == '20201120' or specific_date == '20201125' or specific_date == '20201130':
        groups = load_daily_groups_before_20200922(specific_date)
    elif specific_date < "20210426" or specific_date == "20221109":
        groups = load_daily_groups_before_20210426(specific_date)
    else:
        groups = load_daily_groups(specific_date)
    if groups is None:
        return
    results_dict = {logic_name: [] for logic_name in main_logic_config.keys()}
    if use_multiprocess:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for code, group in tqdm(groups, desc="Submitting tasks"):
                futures[executor.submit(process_group, main_logic_config, group)] = code
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="全局进程池计算"):
                try:
                    res_dict = future.result()
                    for logic_name, res in res_dict.items():
                        results_dict[logic_name].append(res)
                except Exception as e:
                    print(f"处理出错: {e}")
        
    else:

        # for logic_name, cfg in main_logic_config.items():
        #     for _, group in tqdm(groups, desc=f"处理每只股票 - {logic_name}"):
        #         results_dict[logic_name].append(cfg['func'](group))
                
        for code, group in tqdm(groups, desc="处理每只股票", unit="code"):
            for logic_name, cfg in main_logic_config.items():
                results_dict.setdefault(logic_name, []).append(cfg['func'](group))


    merged_data = None
    for logic_name, cfg in main_logic_config.items():
        processed_data = pd.concat(results_dict[logic_name], ignore_index=True)
        result_selected = processed_data[cfg['cols']].rename(columns={'time': 'second'})
        result_selected = rename_columns(result_selected, logic_name)
        if merged_data is None:
            merged_data = result_selected
        else:
            merged_data = pd.merge(merged_data, result_selected, on=['code', 'second'], how='outer')
        del processed_data, result_selected
        gc.collect()
    
    merged_data['code'] = merged_data['code'].astype(str)
    merged_data = (merged_data.replace([np.inf, -np.inf], np.nan).groupby("code", group_keys=False).apply(lambda df: df.ffill().bfill()))
    merged_data = zscore_normalize(merged_data)
    merged_data = format_second_data(merged_data)

    # main_file = os.path.join(base_output, f"{specific_date}.fea")
    # if os.path.exists(main_file) and os.path.getsize(main_file) > 0:
    #     df_main = pd.read_feather(main_file).set_index(['code', 'second'])
    #     merged_data = merged_data.set_index(['code', 'second'])
    #     df_main.update(merged_data)
    #     new_cols = [col for col in merged_data.columns if col not in df_main.columns]
    #     final_merged = pd.concat([df_main, merged_data[new_cols]], axis=1).reset_index()
    # else:
    #     df_main = None
    #     final_merged = merged_data
    
    final_merged = merged_data
    save_path = os.path.join(base_output, f"{specific_date}.fea")
    final_merged = final_merged.sort_values(
        by=["code", "second"],
        key=lambda col: col.astype(int) if col.name == "code" else col
    ).reset_index(drop=True)

    final_merged['code'] = final_merged['code'].astype(int).astype(str).str.zfill(6)
    final_merged.to_feather(save_path)

    del merged_data, final_merged
    gc.collect()



# ============================================================
# 非通用函数，用于特殊字段计算
# ============================================================
@njit
def rolling_zscore(arr, window, eps):
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan

    for i in range(n):
        start = 0 if i < window else i - window + 1
        cnt = 0
        s = 0.0
        
        for j in range(start, i + 1):
            if not np.isnan(arr[j]):
                s += arr[j]
                cnt += 1

        if cnt <= 1 or np.isnan(arr[i]):
            out[i] = np.nan
            continue

        mean = s / cnt

        var = 0.0
        for j in range(start, i + 1):
            if not np.isnan(arr[j]):
                diff = arr[j] - mean
                var += diff * diff

        std = np.sqrt(var / (cnt - 1))
        
        out[i] = (arr[i] - mean) / (std + eps)
    return out

@njit
def rolling_corr(x, y, window):
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan

    for i in range(n):
        start = 0 if i < window else i - window + 1
        cnt = 0
        sx = 0.0
        sy = 0.0
        
        for j in range(start, i + 1):
            if not np.isnan(x[j]) and not np.isnan(y[j]):
                sx += x[j]
                sy += y[j]
                cnt += 1

        if cnt <= 1:
            out[i] = np.nan
            continue

        mx = sx / cnt
        my = sy / cnt

        cov = 0.0
        vx = 0.0
        vy = 0.0
        for j in range(start, i + 1):
            if not np.isnan(x[j]) and not np.isnan(y[j]):
                dx = x[j] - mx
                dy = y[j] - my
                cov += dx * dy
                vx += dx * dx
                vy += dy * dy

        if vx <= EPS or vy <= EPS:
            out[i] = 0
        else:
            out[i] = cov / np.sqrt(vx * vy)

    return out


@njit
def safe_pca_1d(X, n_components=1, clip=1e6, fill_value=0.0):
    n_rows, n_cols = X.shape
    out = np.full((n_rows, n_components), fill_value)

    if n_rows == 0 or n_cols == 0:
        return out

    # clean + clip
    for i in range(n_rows):
        for j in range(n_cols):
            v = X[i, j]
            if not np.isfinite(v):
                v = 0.0
            if v > clip:
                v = clip
            elif v < -clip:
                v = -clip
            X[i, j] = v

    # std
    std = np.zeros(n_cols)
    for j in range(n_cols):
        m = 0.0
        for i in range(n_rows):
            m += X[i, j]
        m /= n_rows

        s = 0.0
        for i in range(n_rows):
            d = X[i, j] - m
            s += d * d
        std[j] = np.sqrt(s / n_rows)

    if np.all(std < 1e-12):
        return out

    # standardize
    Xs = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            if std[j] > 1e-12:
                Xs[i, j] = X[i, j] / std[j]

    # svd
    try:
        U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    except:
        return out

    k = min(n_components, Vt.shape[0])
    for i in range(n_rows):
        for j in range(k):
            out[i, j] = U[i, j] * S[j]

    return out

def _safe_pls_1d(X, y, n_components=2):
    """对单个域进行 PLS 降维，保持与 safe_pca_1d 相同输出格式"""
    X = X.fillna(0).values
    y = y.fillna(0).values.reshape(-1, 1)

    if X.shape[0] < 3:
        return np.zeros((X.shape[0], n_components))

    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    X_scores = pls.x_scores_   # (N, n_components)
    return X_scores

def compute_aroon(high, low, period=25):
    # 找过去 period 根 K 中最近一次最高点和最低点的位置
    rolling_high_idx = high.rolling(period).apply(lambda x: period - 1 - x.argmax(), raw=True)
    rolling_low_idx = low.rolling(period).apply(lambda x: period - 1 - x.argmin(), raw=True)

    aroon_up = 100 * (period - rolling_high_idx) / period
    aroon_down = 100 * (period - rolling_low_idx) / period

    return aroon_up, aroon_down

def calc_entropy(x):
    if len(x) <= 1:
        return np.nan
    p = x.value_counts(normalize=True, sort=False).values
    return -(p * np.log(p + EPS)).sum()

def mutual_information(x, y):
    df = pd.DataFrame({'x': x, 'y': y})
    if len(df) <= 1:
        return np.nan

    pxy = df.value_counts(normalize=True, sort=False)
    px = df['x'].value_counts(normalize=True, sort=False)
    py = df['y'].value_counts(normalize=True, sort=False)
    
    px_aligned = px.reindex(pxy.index.get_level_values('x')).values
    py_aligned = py.reindex(pxy.index.get_level_values('y')).values
    p_vals = pxy.values

    term = p_vals / (px_aligned * py_aligned + EPS)
    return (p_vals * np.log(term + EPS)).sum()

def conditional_entropy(x, y):
    df = pd.DataFrame({'x': x, 'y': y})
    if len(df) <= 1:
        return np.nan

    pxy = df.value_counts(normalize=True, sort=False)
    py = df['y'].value_counts(normalize=True, sort=False)

    py_aligned = py.reindex(pxy.index.get_level_values('y')).values
    p_vals = pxy.values

    term = p_vals / (py_aligned + EPS)
    return -(p_vals * np.log(term + EPS)).sum()



# ============================================================
# 字段计算
# ============================================================
eps = 1e-5
EPS = 1e-9
def process_stock_group_terminal(group):
    aligned_group = align_minute_describe(group)
    window = 20
    k = 2
    
    ma = aligned_group['close'].rolling(window=window,min_periods=1).mean()
    std = aligned_group['close'].rolling(window=window,min_periods=1).std()
    aligned_group['Boll_Middle'] = ma
    aligned_group['Boll_Upper'] = ma + std * k
    aligned_group['Boll_Lower'] = ma - std * k
    aligned_group['Boll_Width'] = (aligned_group['Boll_Upper'] - aligned_group['Boll_Lower']) / aligned_group['close']
    aligned_group['Boll_Position'] = (aligned_group['close'] - aligned_group['Boll_Lower']) / (aligned_group['Boll_Upper'] - aligned_group['Boll_Lower'])
    
    hl = aligned_group['high'] - aligned_group['low']
    hc = (aligned_group['high'] - aligned_group['close'].shift()).bfill().abs()
    lc = (aligned_group['low'] - aligned_group['close'].shift()).bfill().abs()
    tr = np.maximum.reduce([hl.values, hc.values, lc.values])
    tr = pd.Series(tr, index=aligned_group.index)
    atr = tr.ewm(span=window, adjust=False).mean()
    middle = aligned_group['close'].ewm(span=window, adjust=False).mean()
    
    aligned_group['KC_Upper'] = middle + atr * k
    aligned_group['KC_Lower'] = middle - atr * k
    aligned_group['KC_Width'] = (aligned_group['KC_Upper'] - aligned_group['KC_Lower']) / aligned_group['close']
    aligned_group['KC_Position'] = (aligned_group['close'] - aligned_group['KC_Lower']) / (aligned_group['KC_Upper'] - aligned_group['KC_Lower'])
    
    aligned_group['emad'] = aligned_group['close'].ewm(span=9, adjust=False).mean() - aligned_group['close'].ewm(span=25, adjust=False).mean()
    AA = hl/aligned_group['emad']
    aligned_group['Mass'] = AA.ewm(span=25,adjust=False).mean()

    Upper = aligned_group['high'].rolling(window=window, min_periods=1).max()
    Lower = aligned_group['low'].rolling(window=window, min_periods=1).min()
    aligned_group['Donchian_Upper'] = Upper
    aligned_group['Donchian_Lower'] = Lower
    aligned_group['Donchian_Width'] = (aligned_group['Donchian_Upper'] - aligned_group['Donchian_Lower']) / aligned_group['close']
    aligned_group['Donchian_Position'] = np.where(
        (aligned_group['Donchian_Upper'] - aligned_group['Donchian_Lower']) <= EPS,
        0,
        (aligned_group['close'] - aligned_group['Donchian_Lower']) / (aligned_group['Donchian_Upper'] - aligned_group['Donchian_Lower'])
    )

    aligned_group['_tmp1'] = (aligned_group['close'] * 2 + aligned_group['high'] + aligned_group['low']) / 4
    aligned_group['_tmp2'] = aligned_group['_tmp1'].rolling(20,min_periods=1).mean()
    CC = np.abs(aligned_group['_tmp1'] - aligned_group['_tmp2']) / aligned_group['_tmp2']

    values = aligned_group['close'].values
    cc_vals = CC.values

    dd = np.zeros_like(values, dtype=float)
    dd[0] = values[0]  # 初始值 = 第一个close

    for i in range(1, len(values)):
        alpha = cc_vals[i]
        dd[i] = alpha * values[i] + (1 - alpha) * dd[i-1]
        
    aligned_group['alpha_xsii'] = CC
    aligned_group['xsii'] = dd
    
    A = (aligned_group['high'] + aligned_group['low'])/2
    B = (aligned_group['high'].shift(1) + aligned_group['low'].shift(1))/2
    C = aligned_group['high'] - aligned_group['low']
    aligned_group['EM'] = np.where(aligned_group['volume'] <= EPS, 0, (A - B) * C / aligned_group['volume'])
    aligned_group['EMV'] = aligned_group['EM'].rolling(window=14,min_periods=1).sum()
    aligned_group['MAEMV'] = aligned_group['EMV'].rolling(window=9,min_periods=1).mean()
    
    aligned_group['DMA'] = aligned_group['close'].rolling(window=10,min_periods=1).mean() - aligned_group['close'].rolling(window=50,min_periods=1).mean()
    
    aligned_group['midp_1'] = (2 *aligned_group['close'] + aligned_group['high'] + aligned_group['low']) / 4
    aligned_group['midp_2'] = (aligned_group['close'] + aligned_group['high'] + aligned_group['low'] + aligned_group['open']) / 4
    aligned_group['midp_3'] = (aligned_group['close'] + aligned_group['high'] + aligned_group['low']) / 3
    aligned_group['midp_4'] = (aligned_group['high'] + aligned_group['low']) / 2
    
    up_price_1 = aligned_group['high'] - aligned_group['midp_1'].shift(1)
    down_price_1 = aligned_group['midp_1'].shift(1) - aligned_group['low']
    up_price_2 = aligned_group['high'] - aligned_group['midp_2'].shift(1)
    down_price_2 = aligned_group['midp_2'].shift(1) - aligned_group['low']
    up_price_3 = aligned_group['high'] - aligned_group['midp_3'].shift(1)
    down_price_3 = aligned_group['midp_3'].shift(1) - aligned_group['low']
    up_price_4 = aligned_group['high'] - aligned_group['midp_4'].shift(1)
    down_price_4 = aligned_group['midp_4'].shift(1) - aligned_group['low']

    aligned_group['CR1'] = up_price_1.rolling(window=26,min_periods=1).sum() / (down_price_1.rolling(window=26,min_periods=1).sum() + eps) * 100
    aligned_group['CR2'] = up_price_2.rolling(window=26,min_periods=1).sum() / (down_price_2.rolling(window=26,min_periods=1).sum() + eps) * 100
    aligned_group['CR3'] = up_price_3.rolling(window=26,min_periods=1).sum() / (down_price_3.rolling(window=26,min_periods=1).sum() + eps) * 100
    aligned_group['CR4'] = up_price_4.rolling(window=26,min_periods=1).sum() / (down_price_4.rolling(window=26,min_periods=1).sum() + eps) * 100

    aligned_group['code'] = group['code'].iloc[0]
    aligned_group['time'] = second.copy()
    return aligned_group 

def process_stock_group_Kline(group):
    aligned_group = align_minute_describe(group)
    high, low, close, volume = [aligned_group[c] for c in ['high','low','close','volume']]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    
    aligned_group['macd_signal'] = signal
    aligned_group['macd_hist'] = hist
    aligned_group['hist_slope'] = aligned_group['macd_hist'].diff(5)
    aligned_group['macd_strength'] = np.tanh(macd / macd.abs().rolling(60,min_periods=1).max())
    aligned_group['macd_signal_gap'] = np.tanh((macd - signal) / macd.abs().rolling(30,min_periods=1).mean())
    aligned_group['trend_momentum'] = np.tanh((ema12 - ema26) * ema12.diff())
    
    vol_ma12 = aligned_group['volume'].rolling(12, min_periods=1).mean()
    vol_ma26 = aligned_group['volume'].rolling(26, min_periods=1).mean()
    v_dif = vol_ma12 - vol_ma26
    v_signal = v_dif.rolling(9, min_periods=1).mean()
    aligned_group['vmacd'] = v_dif - v_signal
    
    aroon_up, aroon_down = compute_aroon(high, low, period=25)
    aligned_group['aroon_up'] = aroon_up
    aligned_group['aroon_down'] = aroon_down
    aligned_group['aroon_diff'] = aroon_up - aroon_down

    aligned_group['pv_momentum'] = close.pct_change() * (volume / volume.shift(1)).fillna(0)
    aligned_group['pv_corr'] = rolling_corr(close.pct_change().to_numpy(),volume.pct_change().to_numpy(),20)
    
    aligned_group['ema_slope'] = ema12.diff()
    aligned_group['trend_energy'] = aligned_group['ema_slope'] * aligned_group['volume']
    aligned_group['long_short_energy'] = aligned_group['macd_strength'] * aligned_group['pv_corr']
    
    # 价格穿透比率
    aligned_group['tp'] = (high + low + close) / 3
    aligned_group['up_penetration'] = (aligned_group['high'] - aligned_group['tp'].shift(1)).clip(lower=0)
    aligned_group['down_penetration'] = (aligned_group['tp'].shift(1) - aligned_group['low']).clip(lower=0)
    for w in [7,13,26]:
        aligned_group[f'PPR_{w}'] = aligned_group['up_penetration'].rolling(w,min_periods=1).mean() / aligned_group['down_penetration'].rolling(w,min_periods=1).mean()
        aligned_group[f'CVI_{w}'] = (aligned_group['high'] - aligned_group['low']).ewm(span=w, adjust=False).mean().diff(w) / (aligned_group['high'] - aligned_group['low']).ewm(span=w, adjust=False).mean() * 100
        aligned_group[f'PSY_{w}'] = aligned_group['close'].rolling(w,min_periods=1).apply(lambda x: (x.diff() > 0).sum() / w * 100)

    
    ret = (aligned_group['close'] - aligned_group['close'].shift(1))/aligned_group['close'].shift(1)
    ret.fillna(0, inplace=True)
    aligned_group['ret_ts_skew'] = ret.rolling(30,min_periods=1).apply(lambda x: pd.Series(x).skew())
    aligned_group['ret_ts_skew_std'] = aligned_group['ret_ts_skew'].rolling(30,min_periods=1).std()
    aligned_group['ret_ts_kurt'] = ret.rolling(30,min_periods=1).apply(lambda x: pd.Series(x).kurtosis())
    aligned_group['ret_ts_kurt_std'] = aligned_group['ret_ts_kurt'].rolling(30,min_periods=1).std()
    
    aligned_group['signed_vol'] = (aligned_group['close'] > aligned_group['close'].shift(1)) * aligned_group['volume'] - (aligned_group['close'] < aligned_group['close'].shift(1)) * aligned_group['volume']
    aligned_group['EDIT'] = aligned_group['signed_vol'].rolling(20,min_periods=1).sum()
    aligned_group['EPI'] = aligned_group['EDIT'] / (aligned_group['volume'].rolling(20,min_periods=1).sum() + eps)
    
    aligned_group['up_count'] = ((aligned_group['close'] > aligned_group['close'].shift(1)).astype(int)).rolling(30,min_periods=1).sum()
    aligned_group['down_count'] = ((aligned_group['close'] < aligned_group['close'].shift(1)).astype(int)).rolling(30,min_periods=1).sum()
    diff = aligned_group['close'].diff()
    aligned_group['up_std'] = diff.where(diff > 0, 0).rolling(30, min_periods=1).std()
    aligned_group['down_std'] = diff.where(diff < 0, 0).rolling(30, min_periods=1).std()
    aligned_group['rvi'] = 100 * (aligned_group['up_std'] / (aligned_group['up_std'] + aligned_group['down_std'] + eps))
    
    # zbf
    aligned_group['vol_parkinson'] = (1 / (4 * np.log(2))) * (np.log(aligned_group['high'] / aligned_group['low'])) ** 2
    hl_range = aligned_group['high'] - aligned_group['low']
    ema_range = hl_range.ewm(span=30, adjust=False).mean()
    aligned_group['chaikin_vol'] = (ema_range - ema_range.shift(30)) / (ema_range.shift(30) + eps) * 100
        
    aligned_group['code'] = group['code'].iloc[0]
    aligned_group['time'] = second.copy()
    return aligned_group 

def process_stock_group_Klineshape(group):
    aligned_group = align_minute_describe(group)
    
    aligned_group['close_location'] = np.where(
        (aligned_group['high'] - aligned_group['low']) <= EPS,
        0.0,
        ((aligned_group['high'] - aligned_group['close']) - (aligned_group['close'] - aligned_group['low']))/(aligned_group['high'] - aligned_group['low'])
    )
    
    aligned_group['close_loc_before'] = np.where(
        (aligned_group['high'].shift(1) - aligned_group['low'].shift(1)) <= EPS,
        0.0,
        ((aligned_group['high'].shift(1) - aligned_group['close']) - (aligned_group['close'] - aligned_group['low'].shift(1)))/(aligned_group['high'].shift(1) - aligned_group['low'].shift(1))
    )
    
    aligned_group['bullish_candle'] = (aligned_group['high'] - np.maximum(aligned_group['close'],aligned_group['open']))
    aligned_group['bearish_candle'] = (np.minimum(aligned_group['close'],aligned_group['open']) - aligned_group['low'])
    aligned_group['candle_ratio'] = np.where(
        aligned_group['bearish_candle'] <= EPS,
        0.0,
        aligned_group['bullish_candle'] / (aligned_group['bearish_candle'])
    )
    
    aligned_group['candle_skew'] = np.where(
        (aligned_group['high'] - aligned_group['low']) <= EPS,
        0.0,
        (aligned_group['bullish_candle'] - aligned_group['bearish_candle']) / (aligned_group['high'] - aligned_group['low'])
    )
    
    aligned_group['body_size'] = aligned_group['close'] - aligned_group['open']
    aligned_group['body_candle_ratio'] = np.where(
        (aligned_group['bullish_candle'] + aligned_group['bearish_candle']) <= EPS,
        aligned_group['body_size'] / 1e-3,
        aligned_group['body_size'] / (aligned_group['bullish_candle'] + aligned_group['bearish_candle'])
    )

    aligned_group['Doji'] = np.where(
        (aligned_group['high'] - aligned_group['low']) <= EPS,
        0.0,
        (aligned_group['close'] - aligned_group['open']).abs() / (aligned_group['high'] - aligned_group['low'])
    )
    
    aligned_group['overlap_ratio'] = np.where(
        (aligned_group['high'] - aligned_group['low']).shift(1) <= EPS,
        0.0,
        np.maximum(0, np.minimum(aligned_group['high'], aligned_group['high'].shift(1)) - 
                      np.maximum(aligned_group['low'], aligned_group['low'].shift(1))) / (aligned_group['high'] - aligned_group['low']).shift(1)
    )

    aligned_group['gap_ratio'] = np.where(
        aligned_group['close'].shift(1) <= EPS,
        0.0,
        (aligned_group['open'] - aligned_group['close'].shift(1)) / aligned_group['close'].shift(1)
    )
    aligned_group['close_position'] = np.where(
        (aligned_group['high'].shift(1) - aligned_group['low'].shift(1)) <= EPS,
        0.0,
        (aligned_group['close'] - aligned_group['low'].shift(1))/(aligned_group['high'].shift(1) - aligned_group['low'].shift(1))
    )

    aligned_group['open_position'] = np.where(
        (aligned_group['high'].shift(1) - aligned_group['low'].shift(1)) <= EPS,
        0.0,
        (aligned_group['open'] - aligned_group['low'].shift(1))/(aligned_group['high'].shift(1) - aligned_group['low'].shift(1))
    )
    
    # 均线偏离度
    ma = aligned_group['close'].rolling(237, min_periods=1).mean()
    aligned_group['ma_bias'] = (aligned_group['close'] - ma) / (ma + eps)

    aligned_group['mdd'] = (aligned_group['high'] - aligned_group['close'])/aligned_group['high'] * 100
    cum_high = aligned_group['high'].rolling(237,min_periods=1).max()
    aligned_group['cum_mmd'] = (cum_high - aligned_group['close']) / cum_high * 100
    
    aligned_group['code'] = group['code'].iloc[0]
    aligned_group['time'] = second.copy()
    
    return aligned_group

def process_stock_group_pca(group):
    aligned_group = aggregate_lob_with_fill(group, interval='10s')
    aligned_group['midp'] = (aligned_group['bp1'] + aligned_group['sp1']) / 2
    pca_n_components = 2

    # 构造各域字段
    bv_vols, sv_cols = [],[]
    imb = []
    bid_slope, ask_slope= [],[]
    bid_diff, ask_diff = [],[]
    
    # 挂单量
    aligned_group['bv_sum'] = aligned_group[[f'bv{i}' for i in range(1,6)]].sum(axis=1) + eps
    aligned_group['sv_sum'] = aligned_group[[f'sv{i}' for i in range(1,6)]].sum(axis=1) + eps
    
    for i in range(1, 6):
        aligned_group[f'bv_ratio_{i}'] = aligned_group[f'bv{i}'] / aligned_group['bv_sum']
        aligned_group[f'sv_ratio_{i}'] = aligned_group[f'sv{i}'] / aligned_group['sv_sum']
        cum_bv = aligned_group[[f'bv{k}' for k in range(1, i+1)]].sum(axis=1)
        cum_sv = aligned_group[[f'sv{k}' for k in range(1, i+1)]].sum(axis=1)
        aligned_group[f'imb_{i}'] = (cum_bv - cum_sv) / (cum_bv + cum_sv + eps)
        aligned_group[f'bid_slope_{i}'] = (aligned_group[f'bp{i}'] - aligned_group['bp1']) / (cum_bv - aligned_group['bv1'] + eps)
        aligned_group[f'ask_slope_{i}'] = (aligned_group[f'sp{i}'] - aligned_group['sp1']) / (cum_sv - aligned_group['sv1'] + eps)

        bv_vols.append(f'bv_ratio_{i}')
        sv_cols.append(f'sv_ratio_{i}')
        imb.append(f'imb_{i}')
        bid_slope.append(f'bid_slope_{i}')
        ask_slope.append(f'ask_slope_{i}')
        
    for i in range(1, 5):
        aligned_group[f'bid_diff_{i}'] = (aligned_group[f'bv{i}'] - aligned_group[f'bv{i+1}'])/(aligned_group[f'bv{i}'] + aligned_group[f'bv{i+1}'] + eps)
        aligned_group[f'ask_diff_{i}'] = (aligned_group[f'sv{i}'] - aligned_group[f'sv{i+1}'])/(aligned_group[f'sv{i}'] + aligned_group[f'sv{i+1}'] + eps)
        
        bid_diff.append(f'bid_diff_{i}')
        ask_diff.append(f'ask_diff_{i}')
    # 对每个域分别做 PCA
    bv_pcs = safe_pca_1d(aligned_group[bv_vols].to_numpy(np.float64, copy=True),n_components=pca_n_components)
    sv_pcs = safe_pca_1d(aligned_group[sv_cols].to_numpy(np.float64, copy=True),n_components=pca_n_components)
    imb_pcs = safe_pca_1d(aligned_group[imb].to_numpy(np.float64, copy=True),n_components=pca_n_components)
    bid_slope_pcs = safe_pca_1d(aligned_group[bid_slope].to_numpy(np.float64, copy=True),n_components=pca_n_components)
    ask_slope_pcs = safe_pca_1d(aligned_group[ask_slope].to_numpy(np.float64, copy=True),n_components=pca_n_components)
    bid_diff_pcs = safe_pca_1d(aligned_group[bid_diff].to_numpy(np.float64, copy=True),n_components=pca_n_components)
    ask_diff_pcs = safe_pca_1d(aligned_group[ask_diff].to_numpy(np.float64, copy=True),n_components=pca_n_components)

    for k in range(pca_n_components):
        aligned_group[f'Order_bv_{k+1}'] = bv_pcs[:, k]
        aligned_group[f'Order_sv_{k+1}'] = sv_pcs[:, k]
        aligned_group[f'imb_{k+1}'] = imb_pcs[:, k]
        aligned_group[f'bid_slope_{k+1}'] = bid_slope_pcs[:, k]
        aligned_group[f'ask_slope_{k+1}'] = ask_slope_pcs[:, k]
        aligned_group[f'bid_diff_{k+1}'] = bid_diff_pcs[:, k]
        aligned_group[f'ask_diff_{k+1}'] = ask_diff_pcs[:, k]

    result = sample_237(aligned_group).reset_index(drop=True)
    result['time'] = second.copy()
    return result

def process_stock_group_pls(group):
    aligned_group = aggregate_lob_with_fill(group, interval='10s')
    aligned_group['midp'] = (aligned_group['bp1'] + aligned_group['sp1']) / 2
    aligned_group['y'] = aligned_group['midp'].shift(-1) - aligned_group['midp']
    
    pls_n_components = 2

    bv_vols, sv_cols = [], []
    imb = []
    bid_slope, ask_slope = [], []
    bid_diff, ask_diff = [], []

    aligned_group['bv_sum'] = aligned_group[[f'bv{i}' for i in range(1,6)]].sum(axis=1) + eps
    aligned_group['sv_sum'] = aligned_group[[f'sv{i}' for i in range(1,6)]].sum(axis=1) + eps

    for i in range(1,6):
        aligned_group[f'bv_ratio_{i}'] = aligned_group[f'bv{i}'] / aligned_group['bv_sum']
        aligned_group[f'sv_ratio_{i}'] = aligned_group[f'sv{i}'] / aligned_group['sv_sum']

        cum_bv = aligned_group[[f'bv{k}' for k in range(1,i+1)]].sum(axis=1)
        cum_sv = aligned_group[[f'sv{k}' for k in range(1,i+1)]].sum(axis=1)
        aligned_group[f'imb_{i}'] = (cum_bv - cum_sv) / (cum_bv + cum_sv + eps)

        aligned_group[f'bid_slope_{i}'] = (aligned_group[f'bp{i}'] - aligned_group['bp1']) / (cum_bv - aligned_group['bv1'] + eps)
        aligned_group[f'ask_slope_{i}'] = (aligned_group[f'sp{i}'] - aligned_group['sp1']) / (cum_sv - aligned_group['sv1'] + eps)

        bv_vols.append(f'bv_ratio_{i}')
        sv_cols.append(f'sv_ratio_{i}')
        imb.append(f'imb_{i}')
        bid_slope.append(f'bid_slope_{i}')
        ask_slope.append(f'ask_slope_{i}')
        
    for i in range(1,5):
        aligned_group[f'bid_diff_{i}'] = (aligned_group[f'bv{i}'] - aligned_group[f'bv{i+1}']) / (aligned_group[f'bv{i}'] + aligned_group[f'bv{i+1}'] + eps)
        aligned_group[f'ask_diff_{i}'] = (aligned_group[f'sv{i}'] - aligned_group[f'sv{i+1}']) / (aligned_group[f'sv{i}'] + aligned_group[f'sv{i+1}'] + eps)
        
        bid_diff.append(f'bid_diff_{i}')
        ask_diff.append(f'ask_diff_{i}')


    y = aligned_group['y']
    bv_pcs  = _safe_pls_1d(aligned_group[bv_vols],y, n_components=pls_n_components)
    sv_pcs  = _safe_pls_1d(aligned_group[sv_cols],y, n_components=pls_n_components)
    imb_pcs = _safe_pls_1d(aligned_group[imb],y, n_components=pls_n_components)
    bid_slope_pcs = _safe_pls_1d(aligned_group[bid_slope], y, n_components=pls_n_components)
    ask_slope_pcs = _safe_pls_1d(aligned_group[ask_slope], y, n_components=pls_n_components)
    bid_diff_pcs = _safe_pls_1d(aligned_group[bid_diff], y, n_components=pls_n_components)
    ask_diff_pcs = _safe_pls_1d(aligned_group[ask_diff], y, n_components=pls_n_components)

    for k in range(pls_n_components):
        aligned_group[f'Order_bv_{k+1}']   = bv_pcs[:, k]
        aligned_group[f'Order_sv_{k+1}']   = sv_pcs[:, k]
        aligned_group[f'imb_{k+1}']        = imb_pcs[:, k]
        aligned_group[f'bid_slope_{k+1}']  = bid_slope_pcs[:, k]
        aligned_group[f'ask_slope_{k+1}']  = ask_slope_pcs[:, k]
        aligned_group[f'bid_diff_{k+1}'] = bid_diff_pcs[:, k]
        aligned_group[f'ask_diff_{k+1}'] = ask_diff_pcs[:, k]

    result = sample_237(aligned_group).reset_index(drop=True)
    result['time'] = second.copy()
    return result

def process_stock_group_MIC(group):
    df = group.copy()
    minutes = df['time'].dt.floor('1min')
    
    def _minute_features(minute_df):
        amt = minute_df['amt'].to_numpy() # 显式转为 numpy
        if amt.size == 0:
            return None

        q10, q90 = np.quantile(amt, [0.1, 0.9])
        amin = amt.min()
        amax = amt.max()
        denom = amax - amin + eps
        
        price = minute_df['price']
        vol = minute_df['vol']
        bp1 = minute_df['bp1']
        sp1 = minute_df['sp1']
        bv1 = minute_df['bv1']

        mi_price_bp1 = mutual_information(price, bp1)
        mi_vol_bv1 = mutual_information(vol, bv1)
        mi_price_vol = mutual_information(price, vol)
        mi_bp1_sp1   = mutual_information(bp1, sp1)

        return pd.Series({
            'QUA_high': q90,
            'QUA_low': q10,
            'QUA_low_min': (q10 - amin) / denom,
            'QUA_high_low': (q90 - q10) / denom,

            'ENT_price': calc_entropy(price),
            'ENT_vol':   calc_entropy(vol),
            'ENT_bp1':   calc_entropy(bp1),
            'ENT_sp1':   calc_entropy(minute_df['sp1']),
            'ENT_bv1':   calc_entropy(bv1),
            'ENT_sv1':   calc_entropy(minute_df['sv1']),

            'MI_price_bp1': mi_price_bp1,
            'MI_vol_bv1':   mi_vol_bv1,
            'MI_price_vol': mi_price_vol,
            'MI_bp1_sp1':   mi_bp1_sp1,
        })

    aligned_group = (df.groupby(minutes, sort=False).apply(_minute_features).reset_index().rename(columns={'time': 'time'}))

    if aligned_group.empty:
        return aligned_group

    start_day = aligned_group['time'].dt.normalize().iloc[0]

    morning_times = pd.date_range(
        start=start_day + pd.Timedelta(hours=9, minutes=30),
        end=start_day + pd.Timedelta(hours=11, minutes=29),
        freq='1min'
    )

    afternoon_times = pd.date_range(
        start=start_day + pd.Timedelta(hours=13, minutes=0),
        end=start_day + pd.Timedelta(hours=14, minutes=56),
        freq='1min'
    )

    full_times = morning_times.union(afternoon_times)
    full_df = pd.DataFrame({'time': full_times})

    merged = full_df.merge(aligned_group, on='time', how='left')
    merged['code'] = group['code'].iloc[0]
    merged['time'] = second.copy()
    
    return merged


# ============================================================
# 主逻辑配置及通用变量定义
# ============================================================
def expand_with_norms(cols, norms=("z", "mm", "rs", "mad"),keep_original=False):
    expanded = []
    for c in cols:
        if keep_original:
            expanded.append(c)  # 原始列
        for n in norms:
            expanded.append(f"{c}_{n}")
    return expanded

def make_config(func, fields):
    return {
        'func': func,
        'cols': ['code', 'time'] + fields,
        'out_put_cols': ['code', 'second'] + expand_with_norms(
            fields, norms=("z", "mm", "rs", "mad"), keep_original=keep_original
        )
    }

# 是否保留原始列
keep_original = False
main_logic_config = {
    'Terminal':{
        'func': process_stock_group_terminal,
        'cols': ['code', 'time', 'Boll_Upper', 'Boll_Lower','Boll_Width',
                 'Boll_Position', 'KC_Upper', 'KC_Lower', 'KC_Width', 'KC_Position', 'emad',
                 'Mass', 'Donchian_Width', 'Donchian_Position','alpha_xsii', 'xsii',
                 'EM', 'EMV', 'MAEMV', 'DMA', 'CR1', 'CR2', 'CR3', 'CR4']
    },
    'Kline':{
        'func': process_stock_group_Kline,
        'cols': ['code', 'time', 'macd_signal', 'macd_hist', 'hist_slope', 
                 'macd_strength', 'macd_signal_gap','trend_momentum','vmacd',
                 'aroon_up', 'aroon_down', 'aroon_diff', 'pv_momentum', 'pv_corr', 
                 'trend_energy', 'long_short_energy', 'up_penetration','down_penetration',
                 'PPR_7', 'PPR_13', 'PPR_26', 'CVI_7', 'CVI_13', 'CVI_26','PSY_7', 
                 'PSY_13', 'PSY_26', 'ret_ts_skew', 'ret_ts_skew_std', 'ret_ts_kurt', 
                 'ret_ts_kurt_std', 'signed_vol', 'EDIT', 'EPI','up_count', 'down_count',
                 'up_std', 'down_std', 'rvi', 'vol_parkinson', 'chaikin_vol']
    },
    'Klineshape':{
        'func': process_stock_group_Klineshape,
        'cols': ['code', 'time', 'close_location', 'close_loc_before', 
                 'bullish_candle', 'bearish_candle', 'candle_ratio', 
                 'body_candle_ratio','Doji', 'overlap_ratio', 'candle_skew', 'body_size', 'gap_ratio',
                 'close_position', 'open_position', 'ma_bias', 'mdd', 'cum_mmd'],
    },
    'PCA':{
        'func': process_stock_group_pca,
        'cols': ['code', 'time', 'Order_bv_1', 'Order_bv_2', 'Order_sv_1', 'Order_sv_2', 'imb_1', 'imb_2', 'bid_slope_1', 'bid_slope_2',
                 'ask_slope_1', 'ask_slope_2', 'bid_diff_1', 'ask_diff_1', 'bid_diff_2', 'ask_diff_2'],
    },
    'PLS':{
        'func': process_stock_group_pls,
        'cols': ['code', 'time', 'Order_bv_1', 'Order_bv_2', 'Order_sv_1', 'Order_sv_2', 'imb_1', 'imb_2', 'bid_slope_1', 'bid_slope_2',
                 'ask_slope_1', 'ask_slope_2', 'bid_diff_1', 'ask_diff_1', 'bid_diff_2', 'ask_diff_2'],    
    },
    'MIC':{
        'func': process_stock_group_MIC,
        'cols': ['code', 'time', 'QUA_high', 'QUA_high_low',
                 'ENT_price', 'ENT_vol', 'ENT_bp1', 'ENT_sp1', 'ENT_bv1', 'ENT_sv1',
                 'MI_price_bp1', 'MI_vol_bv1', 'MI_price_vol', 'MI_bp1_sp1'],
    }
}


second  = generate_minute_timestamps()
bp_cols = [f'bp{i+1}' for i in range(5)]
sp_cols = [f'sp{i+1}' for i in range(5)]
bv_cols = [f'bv{i+1}' for i in range(5)]
sv_cols = [f'sv{i+1}' for i in range(5)]
LOB_cols = ["bp1", "bv1", "bn1", "bp2", "bv2", "bn2",
            "bp3", "bv3", "bn3", "bp4", "bv4", "bn4",
            "bp5", "bv5", "bn5",
            "sp1", "sv1", "sn1", "sp2", "sv2", "sn2",
            "sp3", "sv3", "sn3", "sp4", "sv4", "sn4",
            "sp5", "sv5", "sn5"]


# ============================================================
# 主程序设定
# ============================================================
if __name__ == '__main__':
    # TODO 需要设定：本批字段的名称，计算的起止日期
    base_data_name = rf'LYT_Test1_Lob'
    start_date = '20220224'
    end_date = '20220224'
    # end_date = '20240930'
    # fea_save_path = rf'{minute_fea_path}/{base_data_name}'
    fea_save_path = rf'/mnt/localdisk/sdb/Stock60sBaseDataUser/user118/Feather/base5_lyt'
    os.makedirs(fea_save_path, exist_ok=True)
    os.chmod(fea_save_path, 0o755)
    trade_date_list = get_trade_days(start_date, end_date)

    use_multiprocess = True
    n_workers = 128
    for specific_date in tqdm(trade_date_list, desc='批量处理日期'):
        process_one_day(specific_date, main_logic_config, fea_save_path, n_workers, use_multiprocess)


