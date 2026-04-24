"""
因子构建模块: 抽象基类 + 注册器 + 批量计算管线。

使用方式:
    1. 继承 FactorBase, 实现 compute() 方法
    2. 用 @FactorRegistry.register 装饰器注册
    3. 调用 FactorRegistry.compute_all() 批量计算并保存

⚠️ 防未来数据:
- compute() 输出的宽表中, T 日因子值只能使用 T-1 及之前的数据
- 框架提供 safe_rolling() 等工具函数辅助实现
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type

import pandas as pd

from Strategy import config
from Strategy.data_io.loader import DailyDataLoader
from Strategy.data_io.saver import save_wide_table
from Strategy.utils.helpers import ensure_tradedate_as_index

logger = logging.getLogger(__name__)


class FactorBase(ABC):
    """
    日频因子抽象基类。

    子类必须实现:
        name: str           -- 因子唯一名称
        compute(daily_data) -- 输入日频数据字典, 返回标准宽表
    """

    name: str = ""

    @abstractmethod
    def compute(self, daily_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算因子。

        Parameters
        ----------
        daily_data : dict
            {field_name: wide_df} 日频数据字典, 如 daily_data['CLOSE_PRICE']

        Returns
        -------
        pd.DataFrame
            标准宽表 (index=TRADE_DATE, columns=股票代码, values=因子值)
            ⚠️ T 日因子值只能使用 T-1 及之前的数据!
        """
        ...

    def compute_and_save(
        self,
        daily_data: Dict[str, pd.DataFrame],
        output_dir: Optional[Path] = None,
    ) -> Path:
        """计算因子并保存为宽表"""
        out = output_dir or config.FACTOR_OUTPUT_DIR
        df = self.compute(daily_data)
        path = save_wide_table(df, out / f"{self.name}.fea")
        logger.info("因子 [%s] 已保存: %s, shape=%s", self.name, path, df.shape)
        return path


class FactorRegistry:
    """因子注册器: 管理所有已注册因子, 支持批量计算"""

    _registry: Dict[str, Type[FactorBase]] = {}

    @classmethod
    def register(cls, factor_cls: Type[FactorBase]) -> Type[FactorBase]:
        """装饰器: 注册因子类"""
        if not factor_cls.name:
            raise ValueError(f"因子类 {factor_cls.__name__} 缺少 name 属性")
        if factor_cls.name in cls._registry:
            logger.warning("因子 [%s] 已注册, 将被覆盖", factor_cls.name)
        cls._registry[factor_cls.name] = factor_cls
        return factor_cls

    @classmethod
    def get(cls, name: str) -> Type[FactorBase]:
        """按名称获取因子类"""
        if name not in cls._registry:
            raise KeyError(f"因子 [{name}] 未注册. 已注册: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_factors(cls) -> List[str]:
        """列出所有已注册因子名"""
        return list(cls._registry.keys())

    @classmethod
    def compute_all(
        cls,
        daily_data: Optional[Dict[str, pd.DataFrame]] = None,
        factor_names: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """
        批量计算并保存因子。

        Parameters
        ----------
        daily_data : dict, optional
            若不提供, 自动加载 config.DAILY_FIELDS 中所有字段
        factor_names : list, optional
            指定要计算的因子名列表, 默认计算全部已注册因子
        output_dir : Path, optional
            输出目录

        Returns
        -------
        dict  {factor_name: saved_path}
        """
        if daily_data is None:
            loader = DailyDataLoader()
            daily_data = loader.load_fields(config.DAILY_FIELDS)
            logger.info("已加载 %d 个日频字段", len(daily_data))

        names = factor_names or cls.list_factors()
        results = {}
        for name in names:
            factor_cls = cls.get(name)
            factor = factor_cls()
            path = factor.compute_and_save(daily_data, output_dir)
            results[name] = path

        logger.info("批量计算完成, 共 %d 个因子", len(results))
        return results


def load_factor(name: str) -> pd.DataFrame:
    """快捷加载已保存的因子宽表"""
    path = config.FACTOR_OUTPUT_DIR / f"{name}.fea"
    if not path.exists():
        raise FileNotFoundError(f"因子文件不存在: {path}")
    try:
        df = pd.read_feather(path)
    except Exception as e:
        err = str(e).lower()
        if "not an arrow file" in err or "arrowinvalid" in type(e).__name__.lower():
            raise RuntimeError(
                f"因子文件损坏或写入未完成（常见为进程被 kill、磁盘满、写入中断）: {path}\n"
                "请删除该 .fea 后重新运行因子计算；之后 save_wide_table 已改为先写 .fea.part 再原子替换，"
                "可降低此类损坏概率。"
            ) from e
        raise
    return ensure_tradedate_as_index(df)


def load_all_factors(names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """批量加载因子, 默认加载 outputs/factors/ 下所有 .fea 文件"""
    factor_dir = config.FACTOR_OUTPUT_DIR
    if names is None:
        names = [f.stem for f in factor_dir.glob("*.fea")]
    return {name: load_factor(name) for name in names}
