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



def align_minute_price_stats(group):
    df = group.copy()
    df = df.set_index('time')

    df['p0_fill'] = df['price'].fillna((df['bp1'] + df['sp1']) / 2)
    df['bv_sum'] = df[bv_cols].sum(axis=1)
    df['sv_sum'] = df[sv_cols].sum(axis=1)
    df['up_tick'] = (df['price'] >= df['sp1']).astype(int)
    df['down_tick'] = (df['price'] <= df['bp1']).astype(int)

    # 每分钟指标计算
    tick_count = df.resample('1min').size()

    price_twap = df.resample('1min')['p0_fill'].mean()
    bp1_twap   = df.resample('1min')['bp1'].mean()
    sp1_twap   = df.resample('1min')['sp1'].mean()
    bvs_twap   = df.resample('1min')['bv_sum'].mean()
    svs_twap   = df.resample('1min')['sv_sum'].mean()
    up_tick_count = df.resample('1min')['up_tick'].sum()
    down_tick_count = df.resample('1min')['down_tick'].sum()

    stats = pd.DataFrame({
        "tick_count": tick_count,
        "p0_twap": price_twap,
        "bp1_twap": bp1_twap,
        "sp1_twap": sp1_twap,
        "bvs_twap": bvs_twap,
        "svs_twap": svs_twap,
        "up_tick_count": up_tick_count,
        "down_tick_count": down_tick_count
    })

    trade_date = group['time'].dt.normalize().iloc[0]
    morning_times   = pd.date_range(trade_date + pd.Timedelta('09:30:00'),
                                    trade_date + pd.Timedelta('11:29:00'), freq='1min')
    afternoon_times = pd.date_range(trade_date + pd.Timedelta('13:00:00'),
                                    trade_date + pd.Timedelta('14:56:00'), freq='1min')
    all_times = morning_times.append(afternoon_times)

    stats = stats.reindex(all_times)

    # 缺失值处理 
    stats['tick_count'] = stats['tick_count'].fillna(0)
    for col in ['p0_twap', 'bp1_twap', 'sp1_twap']:
        stats[col] = stats[col].ffill()

    return stats


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
    # 所有字段按float32保存
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
    df = df.filter((pl.col("time") >= start_filter) & (pl.col("time") <= end_filter))

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
        # result_selected = rename_columns(result_selected, logic_name)
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
    # merged_data = add_tick_fixed_suffix(merged_data)

    main_file = os.path.join(base_output, f"{specific_date}.fea")
    if os.path.exists(main_file) and os.path.getsize(main_file) > 0:
        df_main = pd.read_feather(main_file).set_index(['code', 'second'])
        merged_data = merged_data.set_index(['code', 'second'])
        df_main.update(merged_data)
        new_cols = [col for col in merged_data.columns if col not in df_main.columns]
        final_merged = pd.concat([df_main, merged_data[new_cols]], axis=1).reset_index()
    else:
        df_main = None
        final_merged = merged_data

    save_path = os.path.join(base_output, f"{specific_date}.fea")
    final_merged = final_merged.sort_values(
        by=["code", "second"],
        key=lambda col: col.astype(int) if col.name == "code" else col
    ).reset_index(drop=True)

    final_merged['code'] = final_merged['code'].astype(int).astype(str).str.zfill(6)
    final_merged.to_feather(save_path)

    del merged_data, final_merged, df_main
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

def diff0(arr):
    out = np.empty_like(arr)
    out[0] = 0.0
    out[1:] = arr[1:] - arr[:-1]
    return out

# 对信号进行傅立叶变换，保留幅值最高的前五个频率分量，并通过傅立叶逆变换重构信号
def rolling_fft_5_pows(arr, window):
    n = arr.shape[0]
    result = np.full(n, np.nan)
    result_1 = np.full(n, np.nan)
    for i in range(n):
        left = max(0, i - window + 1)
        data_window = arr[left:i+1]
        if len(data_window) == 0 or np.all(np.isnan(data_window)):
            continue
        fft_data = np.fft.fft(data_window)
        pows = np.abs(fft_data)**2
        topk = min(5, len(pows))
        idx = np.argsort(pows,kind='stable')[-topk:]
        filter_complex = np.zeros_like(fft_data)
        filter_complex[idx] = fft_data[idx]
        filtered = np.fft.ifft(filter_complex)
        result[i] = np.mean(filtered.real)
        result_1[i] = filtered.real[-1]
    return result,result_1

# 计算曲率
def curvature_from_matrix(mat):
    row_sums = mat.sum(axis=1, keepdims=True)
    p = np.divide(mat, row_sums, out=np.zeros_like(mat, dtype=float), where=(row_sums != 0))

    p_left = p[:, :-2]
    p_mid = p[:, 1:-1]
    p_right = p[:, 2:]
    sec = p_left - 2 * p_mid + p_right

    curv_raw = sec.sum(axis=1)
    denom = np.sum(np.abs(sec), axis=1)
    curv_norm = np.divide(curv_raw, denom, out=np.zeros_like(curv_raw), where=(denom != 0))

    return curv_raw, curv_norm

# 计算行相关系数，列相关系数可通过（rolling.corr())获得
# e.g. bp1-bp5 与 sp1-sp5 的相关系数
def compute_row_corr_simple(df, cols_1, cols_2, eps=1e-8):

    bp_mat = df[cols_1].to_numpy(dtype=np.float64)  # (T,k)
    bv_mat = df[cols_2].to_numpy(dtype=np.float64)

    # 行是否全有效（bp 和 bv 对应列都没有 NaN）
    rows_ok = (~np.isnan(bp_mat).any(axis=1)) & (~np.isnan(bv_mat).any(axis=1))
    corr = np.zeros(len(df), dtype=np.float64)

    if rows_ok.any():
        bp_ok = bp_mat[rows_ok]        # (n_ok, k)
        bv_ok = bv_mat[rows_ok]

        bp_center = bp_ok - bp_ok.mean(axis=1, keepdims=True)
        bv_center = bv_ok - bv_ok.mean(axis=1, keepdims=True)

        num = (bp_center * bv_center).sum(axis=1)
        ss_bp = (bp_center**2).sum(axis=1)
        ss_bv = (bv_center**2).sum(axis=1)
        den = np.sqrt(ss_bp * ss_bv)

        # 防止除以 0
        valid_den = den > eps
        corr_ok = np.zeros_like(num)
        corr_ok[valid_den] = num[valid_den] / den[valid_den]

        corr[rows_ok] = corr_ok
    return corr


# ============================================================
# 字段计算
# ============================================================
eps = 1e-5
EPS = 1e-9
def process_stock_group_var(group):
    # 该系列字段来自论文New Evidence of the Marginal Predictive Content of Small and Large Jumps in the Cross-Section，原文5分钟采样生成日频数据，我这边改用1分钟采样生成分钟频数据
    aligned_group = align_minute_describe(group)

    # Jump Variation Calculation
    aligned_group['intraday_log_r'] = np.log(aligned_group['close']/aligned_group['open']) * 100 # 计算日内对数收益率
    aligned_group['r_2'] = aligned_group['intraday_log_r'] ** 2 # 计算对数收益率的平方
    aligned_group['RV'] =  aligned_group['r_2'].cumsum() # 计算日内累积波动率
    aligned_group['V_hat']=(np.abs(aligned_group['intraday_log_r']*aligned_group['intraday_log_r'].shift(1)*aligned_group['intraday_log_r'].shift(2))**(2/3)).fillna(0)
    aligned_group['IV_hat'] = aligned_group['V_hat'].cumsum() # 计算积分波动率（integrated volatility）的估计值
    aligned_group['RVJ'] = np.maximum(aligned_group['RV'] - aligned_group['IV_hat'], 0) # 计算跳跃成分Jump components
    aligned_group['RVC'] = aligned_group['RV'] - aligned_group['RVJ'] # 计算连续成分continuous components
    
    gamma_r = np.abs(aligned_group['intraday_log_r']).quantile(0.99)
    aligned_group['gamma_r'] = gamma_r
    
    # Large and Small Jumps
    aligned_group['RVL'] = (aligned_group['r_2'] * (np.abs(aligned_group['intraday_log_r']) > gamma_r).astype(int)).cumsum()
    aligned_group['RVLJ'] = np.minimum(aligned_group['RVJ'], aligned_group['RVL']) # Large Jumps
    aligned_group['RVSJ'] = aligned_group['RVJ'] - aligned_group['RVLJ'] # Small Jumps
    
    # 考虑upside 和 downside
    aligned_group['RSP'] = (aligned_group['r_2'] * (aligned_group['intraday_log_r'] > 0).astype(int)).cumsum() # Positive RS
    aligned_group['RSN'] = (aligned_group['r_2'] * (aligned_group['intraday_log_r'] < 0).astype(int)).cumsum() # Negative RS
    aligned_group['RVJP'] = np.maximum(aligned_group['RSP'] - 1/2 * aligned_group['IV_hat'], 0) # Positive Jump Variation
    aligned_group['RVJN'] = np.maximum(aligned_group['RSN'] - 1/2 * aligned_group['IV_hat'], 0) # Negative Jump Variation
    aligned_group['SRVJ'] = aligned_group['RVJP'] - aligned_group['RVJN'] # Signed Jump Variation

    # 在考虑upside 和 downside的同时，考虑large和small jump
    aligned_group['RVLP'] = (aligned_group['r_2'] * (aligned_group['intraday_log_r'] > gamma_r).astype(int)).cumsum() # Large Positive
    aligned_group['RVLN'] = (aligned_group['r_2'] * (aligned_group['intraday_log_r'] < -gamma_r).astype(int)).cumsum()
    aligned_group['RVLJP'] = np.minimum(aligned_group['RVJP'], aligned_group['RVLP']) # Large Positive Jumps
    aligned_group['RVLJN'] = np.minimum(aligned_group['RVJN'], aligned_group['RVLN']) # Large Negative Jumps
    aligned_group['RVSJP'] = aligned_group['RVJP'] - aligned_group['RVLJP'] # Small Positive Jumps
    aligned_group['RVSJN'] = aligned_group['RVJN'] - aligned_group['RVLJN'] # Small Negative Jumps

    aligned_group['SRVLJ'] = aligned_group['RVLJP'] - aligned_group['RVLJN'] # Signed Large Jump Variation
    aligned_group['SRVSJ'] = aligned_group['RVSJP'] - aligned_group['RVSJN'] # Signed Small Jump Variation
    
    aligned_group['code'] = group['code'].iloc[0]
    aligned_group['time'] = second.copy()
    return aligned_group

def process_stock_group_vol(group):
    aligned_group= aggregate_lob_with_fill(group,interval='60s')
    aligned_group['bv_sum'] = aligned_group[bv_cols].sum(axis=1)
    aligned_group['sv_sum'] = aligned_group[sv_cols].sum(axis=1)
    
    aligned_group['vol_to_last_bv'] = aligned_group['vol']/(aligned_group['bv_sum'].shift(1))
    aligned_group['vol_to_last_sv'] = aligned_group['vol']/(aligned_group['sv_sum'].shift(1))
    aligned_group['vol_to_bv_diff'] = aligned_group['vol']/np.abs(aligned_group['bv_sum'].diff() + eps)
    aligned_group['vol_to_sv_diff'] = aligned_group['vol']/np.abs(aligned_group['sv_sum'].diff() + eps)

    # 基础成交量特征
    aligned_group['log_vol'] = np.log(aligned_group['vol'] + 1)
    aligned_group['log_bvs'] = np.log(aligned_group['bv_sum'] + 1)
    aligned_group['log_svs'] = np.log(aligned_group['sv_sum'] + 1)

    # 买卖方向（主动买入=1，主动卖出=-1，其余=0），当price为0时无成交，那么vol的值为0，所得基础字段也为0
    direction = ((aligned_group['price'] >= aligned_group['sp1'].shift(1)).astype(int) - (aligned_group['price'] <= aligned_group['bp1'].shift(1)).astype(int)).fillna(0).astype(int)

    aligned_group['signed_vol'] = aligned_group['vol'] * direction
    
    window = 25
    log_vol_arr = aligned_group['log_vol'].to_numpy(dtype=np.float64)
    signed_vol_arr = aligned_group['signed_vol'].to_numpy(dtype=np.float64)
    aligned_group['rolling_fft_log_vol_mean'], aligned_group['rolling_fft_log_vol_last'] = rolling_fft_5_pows(log_vol_arr, window)
    aligned_group['rolling_fft_signed_vol_mean'], aligned_group['rolling_fft_signed_vol_last'] = rolling_fft_5_pows(signed_vol_arr, window)
    
    result = sample_237(aligned_group).reset_index(drop=True)
    result['time'] = second.copy()
    return result    

def process_stock_group_slope(group):
    aligned_group = aggregate_lob_with_fill(group, interval='1s')

    aligned_group['bv_sum'] = aligned_group[bv_cols].sum(axis=1)
    aligned_group['sv_sum'] = aligned_group[sv_cols].sum(axis=1)

    aligned_group['bid_pv_slope'] = (aligned_group['bp5'] - aligned_group['bp1']) / (aligned_group['bv_sum'] - aligned_group['bv1'] + eps)
    aligned_group['ask_pv_slope'] = (aligned_group['sp5'] - aligned_group['sp1']) / (aligned_group['sv_sum'] - aligned_group['sv1'] + eps)
    
    # 这个逻辑比较奇怪，np.log(aligned_group['sp1']) - np.log(aligned_group['bp1'])甚至可能接近于一个常数，但这是原有字段库中的字段，于是在此做记录
    aligned_group['bs_slope'] = (np.log(aligned_group['sp1']) - np.log(aligned_group['bp1']))/(np.log(aligned_group['sv1']) + np.log(aligned_group['bv1']))
    
    aligned_group['bid_diff'] = (aligned_group['bv1'] - aligned_group['bv2'])/(aligned_group['bv1'] + aligned_group['bv2'])
    aligned_group['ask_diff'] = (aligned_group['sv1'] - aligned_group['sv2'])/(aligned_group['sv1'] + aligned_group['sv2'])
    aligned_group['time_diff_slope'] = aligned_group['bid_diff'].diff() - aligned_group['ask_diff'].diff()
    aligned_group['time_diff_slope'] = aligned_group['time_diff_slope'].fillna(0)

    aligned_group['bv_diff'] = aligned_group['bv1'].diff().fillna(0)
    aligned_group['sv_diff'] = aligned_group['sv1'].diff().fillna(0)
    aligned_group['bp_diff'] = aligned_group['bp1'].diff().fillna(0)
    aligned_group['sp_diff'] = aligned_group['sp1'].diff().fillna(0)

    bid_num = aligned_group['bv_diff'] / aligned_group['bv1']
    bid_den = np.where(
        aligned_group['bp_diff'].abs() > EPS,
        aligned_group['bp_diff'] / aligned_group['bp1'],
        (aligned_group['sp1'] - aligned_group['bp1']) / aligned_group['bp1']
    )

    aligned_group['bvp_slope'] = bid_num / (bid_den + EPS)
    
    ask_num = aligned_group['sv_diff'] / aligned_group['sv1']
    ask_den = np.where(
        aligned_group['sp_diff'].abs() > EPS,
        aligned_group['sp_diff'] / aligned_group['sp1'],
        (aligned_group['sp1'] - aligned_group['bp1']) / aligned_group['sp1']
    )

    aligned_group['svp_slope'] = ask_num / (ask_den + EPS)

    aligned_group['std_bvp_slope'] = rolling_zscore(aligned_group['bvp_slope'].to_numpy(),20,eps)
    aligned_group['std_svp_slope'] = rolling_zscore(aligned_group['svp_slope'].to_numpy(),20,eps)

    result = sample_237(aligned_group).reset_index(drop=True)
    result['time'] = second.copy()
    return result

def process_stock_group_imbalance(group):
    aligned_group = aggregate_lob_with_fill(group, interval='10s')
    aligned_group['midp'] = (aligned_group['bp1'] + aligned_group['sp1']) / 2
    ask_weights = [f'aw_{i}' for i in range(1, 6)]
    bid_weights = [f'bw_{i}' for i in range(1, 6)]

    for i in range(5):
        aligned_group[ask_weights[i]] = aligned_group['midp']/(aligned_group[f'sp{i+1}'] - aligned_group['midp'] + eps)
        aligned_group[bid_weights[i]] = aligned_group['midp']/(aligned_group['midp'] - aligned_group[f'bp{i+1}'] + eps)
    # 压力不平衡   
    ask_weight_sum = aligned_group[ask_weights].sum(axis=1)
    aligned_group['press_ask'] = (aligned_group[[f'sv{i+1}' for i in range(5)]] * aligned_group[ask_weights].values).sum(axis=1)/ ask_weight_sum
    bid_weight_sum = aligned_group[bid_weights].sum(axis=1)    
    aligned_group['press_bid'] = (aligned_group[[f'bv{i+1}' for i in range(5)]] * aligned_group[bid_weights].values).sum(axis=1) / bid_weight_sum
    aligned_group['press_imbalance'] = aligned_group['press_bid'] - aligned_group['press_ask']
    
    # 委托量时序差分买卖盘不平衡
    aligned_group['vol_diff_imbalance'] = aligned_group['bv1'].diff().fillna(0) - aligned_group['sv1'].diff().fillna(0)
    aligned_group['delta_vol_bid'] = (aligned_group['bv1'].diff().fillna(0)) * ((aligned_group['bp1'] - aligned_group['bp1'].shift(1)).abs() <= EPS) + aligned_group['bv1'] * (aligned_group['bp1'] > aligned_group['bp1'].shift(1) + EPS) + (aligned_group['bv1'] - aligned_group['bv2'].shift(1)) * (aligned_group['bp1'] < aligned_group['bp1'].shift(1) - EPS)
    aligned_group['delta_vol_ask'] = (aligned_group['sv1'].diff().fillna(0)) * ((aligned_group['sp1'] - aligned_group['sp1'].shift(1)).abs() <= EPS) + aligned_group['sv1'] * (aligned_group['sp1'] < aligned_group['sp1'].shift(1) - EPS) + (aligned_group['sv1'] - aligned_group['sv2'].shift(1)) * (aligned_group['sp1'] > aligned_group['sp1'].shift(1) + EPS)
    aligned_group['delta_vol_imbalance'] = aligned_group['delta_vol_bid'] - aligned_group['delta_vol_ask']
    aligned_group['bid_ratio'] = aligned_group['bv1']/(aligned_group['bv1'] + aligned_group['sv1'])
    aligned_group['pending_ratio_diff_1'] = aligned_group['bid_ratio'].diff().fillna(0)
    aligned_group['pending_ratio_diff_2'] = aligned_group['bid_ratio'].diff(2).fillna(0)
    aligned_group['price_diff_1'] = aligned_group['bp1'].diff()/(aligned_group['bp1'] - aligned_group['bp2'] + eps)
    aligned_group['price_diff_2'] = aligned_group['bp1'].diff(2)/(aligned_group['bp1'] - aligned_group['bp2'] + eps)
    
    aligned_group['ratio_adjust_1'] = aligned_group['pending_ratio_diff_1'] + 0.1 * aligned_group['price_diff_1']
    aligned_group['ratio_adjust_2'] = aligned_group['pending_ratio_diff_2'] + 0.1 * aligned_group['price_diff_2']
    
    # 加权委托价格不平衡
    aligned_group['bv_sum'] = aligned_group[bv_cols].sum(axis=1)
    aligned_group['sv_sum'] = aligned_group[sv_cols].sum(axis=1)
    aligned_group['bid_submit_price'] = ((aligned_group[bp_cols].to_numpy() * aligned_group[bv_cols].to_numpy()).sum(axis=1)/ aligned_group['bv_sum'].to_numpy())
    aligned_group['ask_submit_price'] = ((aligned_group[sp_cols].to_numpy() * aligned_group[sv_cols].to_numpy()).sum(axis=1)/ aligned_group['sv_sum'].to_numpy())
    aligned_group['price_imbalance'] = aligned_group['bid_submit_price'] - aligned_group['ask_submit_price']
    aligned_group['price_to_mid_imbalance'] = (aligned_group['bid_submit_price'] - aligned_group['midp']) - (aligned_group['midp'] - aligned_group['ask_submit_price'])
        
    result = sample_237(aligned_group).reset_index(drop=True)
    result['time'] = second.copy()
    return result

def process_stock_group_shape(group):
    aligned_group = aggregate_lob_with_fill(group, interval='10s')

    # 买方偏度、峰度、JB
    vals_b = aligned_group[bv_cols].to_numpy(dtype=np.float64)
    means_b = vals_b.mean(axis=1, keepdims=True)
    stds_b = vals_b.std(axis=1, ddof=1, keepdims=True)
    stds_b[stds_b == 0] = 1e-9  # 防止除0
    standardized_b = (vals_b - means_b) / stds_b
    skewness_b = np.mean(standardized_b**3, axis=1)
    kurtosis_b = np.mean(standardized_b**4, axis=1) - 3
    n_b = vals_b.shape[1]
    JBtest_b = (skewness_b**2 + (kurtosis_b**2)/4) * n_b / 6

    aligned_group['bv_skew'] = skewness_b
    aligned_group['bv_kurt'] = kurtosis_b
    aligned_group['bv_JBtest'] = JBtest_b

    # 卖方偏度、峰度、JB
    vals_a = aligned_group[sv_cols].to_numpy(dtype=np.float64)
    means_a = vals_a.mean(axis=1, keepdims=True)
    stds_a = vals_a.std(axis=1, ddof=1, keepdims=True)
    stds_a[stds_a == 0] = 1e-9  # 防止除0
    standardized_a = (vals_a - means_a) / stds_a
    skewness_a = np.mean(standardized_a**3, axis=1)
    kurtosis_a = np.mean(standardized_a**4, axis=1) - 3
    n_a = vals_a.shape[1]
    JBtest_a = (skewness_a**2 + (kurtosis_a**2)/4) * n_a / 6

    aligned_group['sv_skew'] = skewness_a
    aligned_group['sv_kurt'] = kurtosis_a
    aligned_group['sv_JBtest'] = JBtest_a

    # 买卖方曲率计算
    bid_mat = vals_b.astype(float)
    ask_mat = vals_a.astype(float)
    bid_curv_raw, bid_curv_norm = curvature_from_matrix(bid_mat)
    ask_curv_raw, ask_curv_norm = curvature_from_matrix(ask_mat)

    aligned_group['bid_curv'] = bid_curv_raw
    aligned_group['ask_curv'] = ask_curv_raw
    aligned_group['bid_curv_norm'] = bid_curv_norm
    aligned_group['ask_curv_norm'] = ask_curv_norm
        
    result = sample_237(aligned_group).reset_index(drop=True)
    result['time'] = second.copy()
    return result

def process_stock_group_corr(group):
    aligned_group = aggregate_lob_with_fill(group, interval='500ms')

    aligned_group['midp'] = (aligned_group['bp1'] + aligned_group['sp1'])/ 2
    aligned_group['bv_sum'] = aligned_group[bv_cols].sum(axis=1)
    aligned_group['sv_sum'] = aligned_group[sv_cols].sum(axis=1)
    
    aligned_group['bp_bv_corr'] = compute_row_corr_simple(aligned_group, cols_1=bp_cols, cols_2=bv_cols)
    aligned_group['sp_sv_corr'] = compute_row_corr_simple(aligned_group, cols_1=sp_cols, cols_2=sv_cols)
    aligned_group['bv_sv_corr'] = compute_row_corr_simple(aligned_group, cols_1=bv_cols, cols_2=sv_cols)
    
    w = 120

    midp   = aligned_group['midp'].to_numpy()
    vol    = aligned_group['vol'].to_numpy()
    bv_sum = aligned_group['bv_sum'].to_numpy()
    sv_sum = aligned_group['sv_sum'].to_numpy()
    bp1    = aligned_group['bp1'].to_numpy()
    sp1    = aligned_group['sp1'].to_numpy()
    bv1    = aligned_group['bv1'].to_numpy()
    sv1    = aligned_group['sv1'].to_numpy()

    # diff
    midp_d = diff0(midp)
    vol_d  = diff0(vol)
    bv_d   = diff0(bv_sum)
    sv_d   = diff0(sv_sum)
    bp1_d  = diff0(bp1)
    sp1_d  = diff0(sp1)
    bv1_d  = diff0(bv1)
    sv1_d  = diff0(sv1)

    aligned_group['midp_bvs_corr'] = rolling_corr(midp, bv_sum, w)
    aligned_group['midp_bvs_diff_corr'] = rolling_corr(midp, bv_d, w)
    aligned_group['midp_svs_corr'] = rolling_corr(midp, sv_sum, w)
    aligned_group['midp_svs_diff_corr'] = rolling_corr(midp, sv_d, w)

    aligned_group['midp_vol_corr'] = rolling_corr(midp, vol, w)
    aligned_group['midp_diff_vol_corr'] = rolling_corr(midp_d, vol, w)

    aligned_group['vol_bvs_diff_corr'] = rolling_corr(vol, bv_d, w)
    aligned_group['vol_svs_diff_corr'] = rolling_corr(vol, sv_d, w)
    aligned_group['vol_bvs_corr'] = rolling_corr(vol, bv_sum, w)
    aligned_group['vol_svs_corr'] = rolling_corr(vol, sv_sum, w)

    aligned_group['bp1_diff_bv1_diff_corr'] = rolling_corr(bp1_d, bv1_d, w)
    aligned_group['sp1_diff_sv1_diff_corr'] = rolling_corr(sp1_d, sv1_d, w)
    aligned_group['bp1_diff_bvs_diff_corr'] = rolling_corr(bp1_d, bv_d, w)
    aligned_group['sp1_diff_svs_diff_corr'] = rolling_corr(sp1_d, sv_d, w)

    corr_cols = [
        'midp_bvs_corr','midp_bvs_diff_corr','midp_svs_corr','midp_svs_diff_corr',
        'midp_vol_corr','midp_diff_vol_corr',
        'vol_bvs_diff_corr','vol_svs_diff_corr','vol_bvs_corr','vol_svs_corr',
        'bp1_diff_bv1_diff_corr','sp1_diff_sv1_diff_corr',
        'bp1_diff_bvs_diff_corr','sp1_diff_svs_diff_corr'
    ]

    aligned_group[corr_cols] = aligned_group[corr_cols].fillna(0.0)
    
    result = sample_237(aligned_group).reset_index(drop=True)
    result['time'] = second.copy()
    return result

def process_stock_group_osod(group):
    aligned_group = align_minute_describe(group)

    # RSV (rolling 9min)
    n = 9
    low_min = aligned_group['low'].rolling(n, min_periods=1).min()
    high_max = aligned_group['high'].rolling(n, min_periods=1).max()
    rsv = (aligned_group['close'] - low_min) / (high_max - low_min + eps) * 100
    rsv = rsv.fillna(50)

    # KDJ
    aligned_group['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    aligned_group['D'] = aligned_group['K'].ewm(alpha=1/3, adjust=False).mean()
    aligned_group['J'] = 3*aligned_group['K'] - 2*aligned_group['D']

    # MA / BIAS
    for w in [6, 12, 24]:
        aligned_group[f'ma_{w}'] = aligned_group['close'].rolling(w, min_periods=1).mean()
        aligned_group[f'bias_{w}'] = (aligned_group['close'] - aligned_group[f'ma_{w}']) / aligned_group[f'ma_{w}'] * 100

    # BR\AR
    prev_close = aligned_group['close'].shift(1).fillna(aligned_group['close'])
    aligned_group['long_strength'] = aligned_group['high'] - prev_close
    aligned_group['short_strength'] = prev_close - aligned_group['low']
    aligned_group['up_push'] = aligned_group['high'] - aligned_group['open']
    aligned_group['down_gravity'] = aligned_group['open'] - aligned_group['low']

    for w in [7,13,26]:
        aligned_group[f'br_{w}'] = aligned_group['long_strength'].rolling(w, min_periods=1).sum() / (aligned_group['short_strength'].rolling(w, min_periods=1).sum() + eps) * 100
        aligned_group[f'ar_{w}'] = aligned_group['up_push'].rolling(w, min_periods=1).sum() / (aligned_group['down_gravity'].rolling(w, min_periods=1).sum() + eps) * 100

    # CCI
    aligned_group['tp'] = (aligned_group['high'] + aligned_group['low'] + aligned_group['close']) / 3
    for w in [5,10,20]:
        aligned_group[f'tp_ma_{w}'] = aligned_group['tp'].rolling(w, min_periods=1).mean()
        aligned_group[f'md_{w}'] = (aligned_group['close'] - aligned_group[f'tp_ma_{w}']).abs().rolling(w, min_periods=1).mean()
        aligned_group[f'cci_{w}'] = (aligned_group['tp'] - aligned_group[f'tp_ma_{w}']) / (aligned_group[f'md_{w}'] + eps) * 100

    aligned_group['code'] = group['code'].iloc[0]
    aligned_group['time'] = second.copy()
    return aligned_group



def process_stock_group_energy(group):
    aligned_group = align_minute_describe(group)
    
    open_, high, low, close, volume = [aligned_group[c] for c in ['open','high','low','close','volume']]
    # OBV calculation
    aligned_group["direction"] = np.sign(close.diff())
    aligned_group["OBV"] = (aligned_group["direction"] * volume).cumsum()

    # True Range calculation
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    aligned_group['TR'] = np.maximum.reduce([tr1, tr2, tr3])

    # Directional Movement calculations
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = high_diff.clip(lower=0)
    minus_dm = low_diff.clip(lower=0)

    # Apply DM selection rule
    mask = plus_dm > minus_dm
    mask1 = plus_dm < minus_dm
    aligned_group['DMP'] = np.where(mask, plus_dm, 0)
    aligned_group['DMN'] = np.where(mask1, minus_dm, 0)

    # Exponential moving averages
    ema_span = 14
    aligned_group['ATR'] = aligned_group['TR'].ewm(span=ema_span, adjust=False).mean()
    smoothed_DMP = aligned_group['DMP'].ewm(span=ema_span, adjust=False).mean()
    smoothed_DMN = aligned_group['DMN'].ewm(span=ema_span, adjust=False).mean()

    # Directional Index calculations
    aligned_group['DIP'] = 100 * smoothed_DMP / aligned_group['ATR']
    aligned_group['DIN'] = 100 * smoothed_DMN / aligned_group['ATR']
    aligned_group['DX'] = 100 * np.abs(aligned_group['DIP'] - aligned_group['DIN']) / (aligned_group['DIP'] + aligned_group['DIN'] + eps)  
    aligned_group['ADX'] = aligned_group['DX'].ewm(span=ema_span, adjust=False).mean()

    # Rolling min/max calculations
    n = 9
    low_min = low.rolling(n, min_periods=1).min()
    high_max = high.rolling(n, min_periods=1).max()

    # RSV calculation
    aligned_group['RSV'] = np.where(high_max != low_min, (close - low_min) / (high_max - low_min + eps) * 100, 50)

    # PR/SI/ASI calculations
    pr = (high_max - low_min) + 0.5 * (high_max - close) + 0.5 * (low_min - close)
    close_shift = close.shift(1, fill_value=close.iloc[0])
    si = 50 * ((close - close_shift) + 0.5*(high_max - close) + 0.5*(low_min - close)) / (pr + eps) * 0.3
    aligned_group['PR'] = pr
    aligned_group['SI'] = si
    aligned_group['ASI'] = si.cumsum()

    # Moving averages
    windows = [3, 6, 12, 24]
    for window in windows:
        aligned_group[f'close_{window}'] = close.rolling(window, min_periods=1).mean()

    aligned_group['BBI'] = aligned_group[[f'close_{w}' for w in windows]].mean(axis=1)

    # RSI calculation
    minute_ret = close.pct_change()
    aligned_group['minute_ret'] = minute_ret
    gain = minute_ret.clip(lower=0)
    loss = -minute_ret.clip(upper=0)

    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    aligned_group['RSI'] = 100 - (100 / (1 + avg_gain / (avg_loss + eps)))
    
    aligned_group['up'] = np.where(aligned_group['minute_ret'] > 0, 1, 0)
    aligned_group['down'] = np.where(aligned_group['minute_ret'] < 0, 1, 0)
    aligned_group['avg_gain'] = gain.rolling(14,min_periods=1).sum()/(aligned_group['up'].rolling(14,min_periods=1).sum() + eps)
    aligned_group['avg_loss'] = loss.rolling(14,min_periods=1).sum()/(aligned_group['down'].rolling(14,min_periods=1).sum() + eps)
    aligned_group['RSII'] = aligned_group['avg_gain']/(aligned_group['avg_gain'] + aligned_group['avg_loss'])*100
    

    # Money Flow Index calculation
    aligned_group['tp'] = (high + low + close) / 3
    aligned_group['money_flow'] = aligned_group['tp'] * volume
    aligned_group['money_flow_ratio'] = (aligned_group['money_flow']*aligned_group['up']).rolling(14,min_periods=1).sum()/((aligned_group['money_flow']*aligned_group['down']).rolling(14,min_periods=1).sum() + eps)
    aligned_group['MFI'] = 100 - 100 / (1 + aligned_group['money_flow_ratio'])
    
    aligned_group['VR'] = (aligned_group['volume'] * aligned_group['up']).rolling(12,min_periods=1).sum() / ((aligned_group['volume'] * aligned_group['down']).rolling(12,min_periods=1).sum() + eps) * 100

    aligned_group['code'] = group['code'].iloc[0]
    aligned_group['time'] = second.copy()
    return aligned_group


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
    'Var': {
        'func': process_stock_group_var,
        'cols': ['code', 'time', 'RV', 'IV_hat', 'RVJ', 'RVC', 'RVSJ', 'RSP', 
                 'RSN', 'RVJP', 'RVJN', 'SRVJ', 'RVSJP', 'RVSJN', 'SRVSJ'],  # 替换为真实列
    },
    'Vol': {
        'func': process_stock_group_vol,
        'cols': ['code', 'time', 'vol_to_last_bv','vol_to_last_sv','vol_to_bv_diff','vol_to_sv_diff',
                'log_bvs','log_svs','rolling_fft_log_vol_mean',
                'rolling_fft_log_vol_last','rolling_fft_signed_vol_last']
    },
    'Slope': {
        'func': process_stock_group_slope,
        'cols': ['code', 'time', 'bid_pv_slope','ask_pv_slope','bs_slope','bid_diff','ask_diff',
                'time_diff_slope','std_bvp_slope','std_svp_slope'],
    },
    'Imbalance': {
        'func': process_stock_group_imbalance, 
        'cols': ['code', 'time', 'press_ask','press_bid','press_imbalance','vol_diff_imbalance',
                'bid_ratio','pending_ratio_diff_1','pending_ratio_diff_2',
                'bid_submit_price','ask_submit_price','ratio_adjust_1','ratio_adjust_2',
                'price_imbalance','price_to_mid_imbalance',
                'delta_vol_bid','delta_vol_ask','delta_vol_imbalance'],
    },
    'Shape': {
        'func': process_stock_group_shape,
        'cols': ['code', 'time', 'bv_skew', 'bv_kurt', 'bv_JBtest', 'sv_skew', 'sv_kurt', 
                'sv_JBtest', 'bid_curv', 'ask_curv', 'bid_curv_norm', 'ask_curv_norm'],
    },
    'Corr': {
        'func': process_stock_group_corr,
        'cols': ['code','time','bp_bv_corr','sp_sv_corr','bv_sv_corr','midp_bvs_corr',
                'midp_bvs_diff_corr','midp_svs_corr','midp_svs_diff_corr',
                'midp_vol_corr','midp_diff_vol_corr','vol_bvs_diff_corr',
                'vol_svs_diff_corr','vol_bvs_corr','vol_svs_corr',
                'bp1_diff_bv1_diff_corr','sp1_diff_sv1_diff_corr',
                'bp1_diff_bvs_diff_corr','sp1_diff_svs_diff_corr'],
    },
    'Osod':{
        'func': process_stock_group_osod,
        'cols': ['code','time','K','D','J','ma_6','ma_12','ma_24',
                'bias_6','bias_12','bias_24','br_7','br_13','br_26',
                'ar_7','ar_13','ar_26','tp','tp_ma_5','tp_ma_10','tp_ma_20',
                'md_5','md_10','md_20','cci_5','cci_10','cci_20']
    },
    'Energy':{
        'func': process_stock_group_energy,
        'cols':['code','time', 'OBV', 'ATR', 'DIP', 'DIN', 'DX', 'ADX', 'ASI', 'BBI','avg_gain','avg_loss','RSI', 'RSII','MFI', 'VR']
    },
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
    fea_save_path = rf'/mnt/localdisk/sdb/Stock60sBaseDataUser/user118/Feather/base0_lyt'
    os.makedirs(fea_save_path, exist_ok=True)
    os.chmod(fea_save_path, 0o755)
    trade_date_list = get_trade_days(start_date, end_date)

    use_multiprocess = True
    n_workers = 64
    for specific_date in tqdm(trade_date_list, desc='批量处理日期'):
        process_one_day(specific_date, main_logic_config, fea_save_path, n_workers, use_multiprocess)


