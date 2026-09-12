import pandas as pd
import asyncio
import aiohttp
import json
import logging
import itertools
from pathlib import Path

from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    before_sleep_log
)
from tqdm import tqdm
from aiolimiter import AsyncLimiter


# =========================================================
# 1. 日志配置
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# 2. API 配置
# =========================================================

# 填入 SiliconFlow API Key
# 如果有多个 Key，可以：
# API_KEYS = ["sk-xxx1", "sk-xxx2", "sk-xxx3"]
API_KEYS = [
    ""
]

API_URL = "https://api.siliconflow.cn/v1/chat/completions"

MODEL_NAME = "deepseek-ai/DeepSeek-V3.1-Terminus"


# =========================================================
# 3. 处理参数
# =========================================================

# API 单次请求最多重试次数
MAX_RETRIES = 3
# 单次请求超时时间（秒）
TIMEOUT_SECONDS = 60
# 每批读取多少条投诉
BATCH_SIZE = 30
# 全局最大并发请求数
CONCURRENT_REQUESTS = 5
# API 请求失败后的重试等待时间
RETRY_DELAY = 5
# 每个批次之间的等待时间
BATCH_DELAY = 2
# 每个 API Key 每分钟最大请求数
PER_KEY_RPM = 50


# =========================================================
# 4. 文件路径
# =========================================================

input_file = r""
output_file = r""
input_sheet = ""


# =========================================================
# 5. Prompt
# =========================================================

PROMPT_TEMPLATE = """
你是一名汽车领域资深产品安全工程师。你的任务是从汽车用户投诉文本中识别其中明确描述的产品缺陷，并提取缺陷对应的零部件、失效模式以及已经实际发生的后果。

【核心任务】

一条投诉可能：

- 不包含能够明确识别的产品缺陷；
- 包含1个产品缺陷；
- 包含多个产品缺陷。

你必须根据投诉文本本身判断其中包含多少个独立缺陷，不允许预先限制缺陷数量。

每个识别出的缺陷使用以下三个字段表示：

1. "零部件"
2. "失效模式"
3. "已发生后果"


【缺陷识别与拆分规则】

1. 所有提取结果必须严格来源于投诉原文，不得根据汽车专业知识推测、补充或创造原文没有明确描述的信息。

2. 一个缺陷对象原则上表示一个独立的：

   “零部件—失效模式—已发生后果”

   组合。

3. 如果投诉中明确描述了多个不同零部件存在不同问题，应分别提取为多个缺陷对象。

4. 如果同一个零部件存在多个明显不同、相互独立的失效模式，可以分别提取。

5. 不得机械拆分。
   如果多个描述只是同一个故障现象的不同表述、连续表现或同一失效模式的具体症状，应尽量合并为一个缺陷，而不是人为增加缺陷数量。

6. “零部件”应提取投诉文本中能够明确识别的汽车零部件、系统或功能模块名称，并尽可能简洁。
   例如：
   "动力电池"
   "制动系统"
   "转向系统"
   "车机系统"
   "空调系统"
   "充电系统"

7. “失效模式”描述该零部件实际发生了什么异常，应尽可能简洁。
   例如：
   "无法充电"
   "异响"
   "漏油"
   "黑屏"
   "制动失效"
   "动力中断"

8. “已发生后果”只能填写投诉文本中已经明确实际发生的结果。

   可以提取：
   "车辆无法启动"
   "车辆失去动力"
   "发生碰撞"
   "无法正常使用"
   "车辆抛锚"

   不得把以下推测性、可能性或风险性描述作为已发生后果：
   "可能发生事故"
   "存在安全隐患"
   "可能导致起火"
   "容易造成人员伤亡"

9. 如果某个缺陷没有明确描述已经发生的后果，则：
   "已发生后果": ""

10. 如果零部件和失效模式能够明确识别，但投诉没有明确说明具体后果，可以正常提取该缺陷，并将后果留空。

11. 如果文本中没有能够明确识别的产品缺陷，则返回空的"缺陷列表"。

12. 不得因为投诉中提到了维修、4S店、厂家处理、消费者情绪、服务态度、索赔、宣传承诺、配置争议等内容，就将这些内容识别为产品缺陷。只有原文明确描述车辆零部件、系统或功能发生实际异常时，才能识别为产品缺陷。

13. 输出内容必须尽可能简洁，不要直接复制大段投诉原文。

14. 对于同一个投诉，不要输出完全重复的缺陷。

15. 少样本示例仅用于学习缺陷识别、拆分、合并和字段填写规则。处理新的投诉时，必须完全依据新的投诉原文进行判断，不得将示例中的零部件、失效模式或后果迁移到新投诉中。


【少样本示例】

示例1

详细文本：
"右后减振器仅仅行驶1.7万公里就出现漏油，车辆内饰异响较为严重。"

提取结果：
{{
  "缺陷列表": [
    {{
      "零部件": "右后减振器",
      "失效模式": "漏油",
      "已发生后果": ""
    }},
    {{
      "零部件": "内饰",
      "失效模式": "异响",
      "已发生后果": ""
    }}
  ]
}}


示例2

详细文本：
"车机卡顿。车机推送1.0.4版本。更新完之后黑屏死机，无法正常使用。新车一年中控内饰掉漆掉皮。"

提取结果：
{{
  "缺陷列表": [
    {{
      "零部件": "车机系统",
      "失效模式": "卡顿、黑屏死机",
      "已发生后果": "无法正常使用"
    }},
    {{
      "零部件": "中控内饰",
      "失效模式": "掉漆掉皮",
      "已发生后果": ""
    }}
  ]
}}


示例3

详细文本：
"蒸发箱铝氧化脱落铝粉，买车就有，一直没有留意，说我超保，严重质量问题。"

提取结果：
{{
  "缺陷列表": [
    {{
      "零部件": "蒸发箱",
      "失效模式": "铝氧化脱落铝粉",
      "已发生后果": ""
    }}
  ]
}}


示例4

详细文本：
"厂家承诺支持发动机全速域直驱，实际使用中并不没有此功能，经4S检测车辆没有质量问题是厂家欺骗消费者一直回避不提出解决方案"

提取结果：
{{
  "缺陷列表": []
}}


示例5

详细文本：
"从购车起到目前使用2年，车内异味问题相当严重，期间在比亚迪售后也处理过多次，一直没有效果，内饰的气味一直存在且没有减轻。空调气味越来越重。"

提取结果：
{{
  "缺陷列表": [
    {{
      "零部件": "内饰",
      "失效模式": "异味",
      "已发生后果": ""
    }},
    {{
      "零部件": "空调系统",
      "失效模式": "异味",
      "已发生后果": ""
    }}
  ]
}}


【待处理投诉详细文本】

{record_info}


【JSON输出要求】

请严格按照上述规则和少样本示例，对“待处理投诉详细文本”进行分析。

必须严格输出一个合法JSON对象。

禁止输出解释、分析、Markdown代码块或任何JSON之外的文字。

严格采用以下结构：

{{
  "缺陷列表": [
    {{
      "零部件": "",
      "失效模式": "",
      "已发生后果": ""
    }}
  ]
}}

如果存在多个缺陷，则按照它们在原文中的出现顺序依次输出。

如果没有能够明确识别的产品缺陷，则必须输出：

{{
  "缺陷列表": []
}}
"""


# =========================================================
# 6. 基础辅助函数
# =========================================================

def clean_text(value):
    """
    清洗Excel中的文本字段。
    防止NaN、None等被发送给LLM。
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def clean_json_content(content):
    """
    清洗LLM可能额外返回的Markdown代码块。
    正常JSON Mode下一般不会出现，但保留作为兜底。
    """

    if content is None:
        return ""

    content = content.strip()

    if content.startswith("```json"):
        content = content[7:].strip()
    elif content.startswith("```"):
        content = content[3:].strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    return content


def normalize_defects(defects):
    """
    对LLM输出的缺陷列表进行规范化：
    1. 保证字段存在
    2. 清除None
    3. 去除前后空格
    4. 删除完全空的缺陷
    5. 删除完全重复的缺陷
    """

    normalized_results = []

    seen = set()

    for item in defects:

        if not isinstance(item, dict):
            continue

        component = clean_text(
            item.get("零部件")
            or item.get("零部件信息")
            or item.get("部件")
            or ""
        )

        failure_mode = clean_text(
            item.get("失效模式")
            or item.get("失效模式描述")
            or item.get("故障模式")
            or ""
        )

        consequence = clean_text(
            item.get("已发生后果")
            or item.get("后果")
            or item.get("实际后果")
            or ""
        )

        # 三个字段全部为空，则不是有效缺陷
        if not component and not failure_mode and not consequence:
            continue

        # 用三元组去重
        defect_key = (
            component,
            failure_mode,
            consequence
        )

        if defect_key in seen:
            continue

        seen.add(defect_key)

        normalized_results.append({
            "零部件": component,
            "失效模式": failure_mode,
            "已发生后果": consequence
        })

    return normalized_results


# =========================================================
# 7. 调用LLM
# =========================================================

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(RETRY_DELAY),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def call_llm_api(
        session,
        complaint_text,
        api_key
):
    """
    调用LLM处理单条投诉。

    如果发生：
    - HTTP错误
    - 超时
    - JSON解析错误
    - JSON结构错误

    均抛出异常，从而触发tenacity自动重试。
    """

    complaint_text = clean_text(complaint_text)

    # 空投诉不调用API
    if not complaint_text:
        return []

    prompt = PROMPT_TEMPLATE.format(
        record_info=complaint_text
    )

    payload = {
        "model": MODEL_NAME,

        "messages": [
            {
                "role": "system",
                "content": (
                    "你是汽车产品缺陷信息抽取模型。"
                    "必须严格依据投诉原文识别0个、1个或多个独立产品缺陷，"
                    "不得推测文本中没有明确描述的信息。"
                    "只能输出合法JSON对象，"
                    "所有缺陷必须放入“缺陷列表”数组。"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        "max_tokens": 2048,

        # 信息抽取任务，尽量降低随机性
        "temperature": 0.0,

        "top_p": 0.7,

        # 不建议在信息抽取任务中设置较高frequency_penalty
        "frequency_penalty": 0.0,

        "response_format": {
            "type": "json_object"
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:

        async with session.post(
                API_URL,
                headers=headers,
                json=payload
        ) as response:

            response_text = await response.text()

            # -------------------------------------------------
            # HTTP错误
            # -------------------------------------------------

            if response.status != 200:

                error_msg = (
                    f"API请求失败，HTTP状态码={response.status}，"
                    f"返回内容={response_text[:500]}"
                )

                logger.error(error_msg)

                raise RuntimeError(error_msg)

            # -------------------------------------------------
            # 解析API外层响应
            # -------------------------------------------------

            try:
                response_data = json.loads(response_text)

            except json.JSONDecodeError as e:

                logger.error(
                    f"API外层响应不是合法JSON：{response_text[:500]}"
                )

                raise ValueError(
                    "API外层响应JSON解析失败"
                ) from e

            # -------------------------------------------------
            # 提取模型输出
            # -------------------------------------------------

            try:

                content = response_data[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ]

            except (
                KeyError,
                IndexError,
                TypeError
            ) as e:

                logger.error(
                    f"API返回结构异常："
                    f"{response_text[:500]}"
                )

                raise ValueError(
                    "API响应中不存在有效choices/message/content"
                ) from e

            logger.debug(
                f"LLM原始输出：{str(content)[:500]}"
            )

            # -------------------------------------------------
            # 清除可能出现的markdown代码块
            # -------------------------------------------------

            content = clean_json_content(content)

            if not content:

                raise ValueError(
                    "LLM返回内容为空"
                )

            # -------------------------------------------------
            # 解析模型生成的JSON
            # -------------------------------------------------

            try:

                result = json.loads(content)

            except json.JSONDecodeError as e:

                logger.error(
                    f"LLM JSON解析失败：{content[:500]}"
                )

                # 必须raise，
                # 这样tenacity才能重新请求
                raise ValueError(
                    "LLM返回内容不是合法JSON"
                ) from e

            # -------------------------------------------------
            # 检查最外层结构
            # -------------------------------------------------

            if not isinstance(result, dict):

                logger.error(
                    f"LLM最外层结果不是JSON对象："
                    f"{type(result)}"
                )

                raise ValueError(
                    "LLM最外层必须为JSON对象"
                )

            # -------------------------------------------------
            # 检查缺陷列表
            # -------------------------------------------------

            if "缺陷列表" not in result:

                logger.error(
                    f"LLM返回结果缺少“缺陷列表”："
                    f"{result}"
                )

                raise ValueError(
                    "LLM输出缺少“缺陷列表”字段"
                )

            defects = result["缺陷列表"]

            if defects is None:
                defects = []

            if not isinstance(defects, list):

                logger.error(
                    f"“缺陷列表”不是数组："
                    f"{type(defects)}"
                )

                raise ValueError(
                    "“缺陷列表”必须为JSON数组"
                )

            # -------------------------------------------------
            # 规范化并返回
            # -------------------------------------------------

            return normalize_defects(defects)

    except asyncio.TimeoutError:

        logger.warning(
            "LLM请求超时"
        )

        raise

    except aiohttp.ClientError as e:

        logger.warning(
            f"网络请求失败：{str(e)}"
        )

        raise


# =========================================================
# 8. 处理单条投诉
# =========================================================

async def process_record(
        session,
        record,
        api_key,
        limiter,
        semaphore
):
    """
    处理一条投诉。

    一条投诉最终可能生成：
    1. 多条缺陷记录
    2. 一条“未识别缺陷”记录
    3. 一条“处理失败”记录
    """

    record_id = clean_text(
        record.get("投诉编号", "")
    )

    desc = clean_text(
        record.get("详细文本", "")
    )

    # -----------------------------------------------------
    # 空文本
    # -----------------------------------------------------

    if not desc:

        logger.warning(
            f"投诉 {record_id} 的详细文本为空"
        )

        return [{
            "投诉编号": record_id,
            "详细文本": "",
            "缺陷序号": "",
            "零部件": "",
            "失效模式": "",
            "已发生后果": "",
            "处理状态": "原始文本为空"
        }]

    try:

        # 全局并发控制
        async with semaphore:

            # 单API Key RPM控制
            async with limiter:

                defects = await call_llm_api(
                    session,
                    desc,
                    api_key
                )

        # -------------------------------------------------
        # 正常完成，但没有识别出缺陷
        # -------------------------------------------------

        if len(defects) == 0:

            logger.info(
                f"投诉 {record_id}：未识别出明确产品缺陷"
            )

            return [{
                "投诉编号": record_id,
                "详细文本": desc,
                "缺陷序号": "",
                "零部件": "",
                "失效模式": "",
                "已发生后果": "",
                "处理状态": "成功-未识别缺陷"
            }]

        # -------------------------------------------------
        # 一个投诉对应多个缺陷
        # -------------------------------------------------

        output_rows = []

        for defect_index, item in enumerate(
                defects,
                start=1
        ):

            output_rows.append({

                "投诉编号": record_id,

                "详细文本": desc,

                # 方便后续确认：
                # 同一个投诉的第几个缺陷
                "缺陷序号": defect_index,

                "零部件": item.get(
                    "零部件",
                    ""
                ),

                "失效模式": item.get(
                    "失效模式",
                    ""
                ),

                "已发生后果": item.get(
                    "已发生后果",
                    ""
                ),

                "处理状态": "成功"
            })

        logger.info(
            f"投诉 {record_id}："
            f"识别出 {len(output_rows)} 个缺陷"
        )

        return output_rows

    except Exception as e:

        # 已经经过MAX_RETRIES次尝试
        # 最终仍然失败

        logger.error(
            f"投诉 {record_id} 处理失败：{str(e)}"
        )

        # 保留这条投诉，不能直接丢失
        return [{
            "投诉编号": record_id,
            "详细文本": desc,
            "缺陷序号": "",
            "零部件": "",
            "失效模式": "",
            "已发生后果": "",
            "处理状态": "处理失败"
        }]


# =========================================================
# 9. 批量处理
# =========================================================

async def process_batch(
        session,
        batch,
        key_cycle,
        limiters,
        semaphore
):
    """
    同时处理一个批次。
    """

    tasks = []

    for record in batch:

        # 多API Key循环分配
        api_key = next(key_cycle)

        limiter = limiters[api_key]

        task = process_record(
            session=session,
            record=record,
            api_key=api_key,
            limiter=limiter,
            semaphore=semaphore
        )

        tasks.append(task)

    results = await asyncio.gather(
        *tasks
    )

    return results


# =========================================================
# 10. 保存Excel
# =========================================================

def save_results(
        all_results,
        output_path
):
    """
    保存全部结果。
    """

    columns = [
        "投诉编号",
        "详细文本",
        "缺陷序号",
        "零部件",
        "失效模式",
        "已发生后果",
        "处理状态"
    ]

    df_out = pd.DataFrame(
        all_results,
        columns=columns
    )

    # -----------------------------------------------------
    # 仅针对“成功识别的缺陷”做完全重复检查
    #
    # 理论上normalize_defects已经在单条投诉内部去重，
    # 此处作为第二层保险。
    # -----------------------------------------------------

    if not df_out.empty:

        df_out = df_out.drop_duplicates(
            subset=[
                "投诉编号",
                "详细文本",
                "缺陷序号",
                "零部件",
                "失效模式",
                "已发生后果",
                "处理状态"
            ],
            keep="first"
        )

    # 创建输出目录
    output_parent = Path(
        output_path
    ).parent

    output_parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 保存Excel
    df_out.to_excel(
        output_path,
        index=False,
        engine="openpyxl"
    )

    return df_out


# =========================================================
# 11. 主函数
# =========================================================

async def main():

    # -----------------------------------------------------
    # 检查API Key
    # -----------------------------------------------------

    valid_api_keys = [
        key.strip()
        for key in API_KEYS
        if key and key.strip()
    ]

    if not valid_api_keys:

        raise ValueError(
            "API_KEYS为空。请先填写有效的SiliconFlow API Key。"
        )

    # -----------------------------------------------------
    # 检查输入文件
    # -----------------------------------------------------

    if not Path(input_file).exists():

        raise FileNotFoundError(
            f"找不到输入文件：{input_file}"
        )

    # -----------------------------------------------------
    # 读取Excel
    # -----------------------------------------------------

    logger.info(
        f"正在读取文件：{input_file}"
    )

    df = pd.read_excel(
        input_file,
        sheet_name=input_sheet,
        dtype={
            "投诉编号": str,
            "详细文本": str
        }
    )

    # -----------------------------------------------------
    # 检查必须字段
    # -----------------------------------------------------

    required_columns = {
        "投诉编号",
        "详细文本"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"输入文件缺少以下字段："
            f"{missing_columns}"
        )

    # 只保留需要的数据
    df = df[
        [
            "投诉编号",
            "详细文本"
        ]
    ].copy()

    # 清除Excel NaN
    df["投诉编号"] = (
        df["投诉编号"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["详细文本"] = (
        df["详细文本"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    total_records = len(df)

    logger.info(
        f"成功读取 {total_records} 条投诉记录"
    )

    # -----------------------------------------------------
    # 初始化控制器
    # -----------------------------------------------------

    semaphore = asyncio.Semaphore(
        CONCURRENT_REQUESTS
    )

    # 每个API Key独立限流
    limiters = {
        key: AsyncLimiter(
            PER_KEY_RPM,
            60
        )
        for key in valid_api_keys
    }

    # 全局循环，不在每个batch重新开始
    key_cycle = itertools.cycle(
        valid_api_keys
    )

    all_results = []

    # -----------------------------------------------------
    # HTTP Session配置
    # -----------------------------------------------------

    timeout = aiohttp.ClientTimeout(
        total=TIMEOUT_SECONDS
    )

    connector = aiohttp.TCPConnector(
        limit=CONCURRENT_REQUESTS * 2
    )

    # -----------------------------------------------------
    # 开始处理
    # -----------------------------------------------------

    async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
    ) as session:

        with tqdm(
                total=total_records,
                desc="投诉缺陷抽取进度",
                unit="条"
        ) as pbar:

            total_batches = (
                total_records
                + BATCH_SIZE
                - 1
            ) // BATCH_SIZE

            for batch_index, start in enumerate(
                    range(
                        0,
                        total_records,
                        BATCH_SIZE
                    ),
                    start=1
            ):

                end = min(
                    start + BATCH_SIZE,
                    total_records
                )

                batch = (
                    df.iloc[start:end]
                    .to_dict("records")
                )

                logger.info(
                    f"开始处理批次 "
                    f"{batch_index}/{total_batches}，"
                    f"投诉 {start + 1}-{end}"
                )

                batch_results = await process_batch(
                    session=session,
                    batch=batch,
                    key_cycle=key_cycle,
                    limiters=limiters,
                    semaphore=semaphore
                )

                # 一条投诉可能对应多行
                for rows in batch_results:
                    all_results.extend(rows)

                # -------------------------------------------------
                # 每个批次完成后立即保存
                #
                # 即使中途程序停止，
                # 已经处理完的批次也不会完全丢失。
                # -------------------------------------------------

                df_saved = save_results(
                    all_results,
                    output_file
                )

                logger.info(
                    f"批次 {batch_index} 完成，"
                    f"当前已保存 {len(df_saved)} 行结果"
                )

                pbar.update(
                    len(batch)
                )

                # 批次之间等待
                if end < total_records:

                    await asyncio.sleep(
                        BATCH_DELAY
                    )

    # -----------------------------------------------------
    # 最终保存
    # -----------------------------------------------------

    final_df = save_results(
        all_results,
        output_file
    )

    # -----------------------------------------------------
    # 输出统计
    # -----------------------------------------------------

    success_defects = final_df[
        final_df["处理状态"] == "成功"
    ]

    no_defect_records = final_df[
        final_df["处理状态"]
        == "成功-未识别缺陷"
    ]

    failed_records = final_df[
        final_df["处理状态"]
        == "处理失败"
    ]

    empty_records = final_df[
        final_df["处理状态"]
        == "原始文本为空"
    ]

    logger.info(
        "=" * 60
    )

    logger.info(
        "全部处理完成"
    )

    logger.info(
        f"原始投诉数：{total_records}"
    )

    logger.info(
        f"识别出的缺陷总数："
        f"{len(success_defects)}"
    )

    logger.info(
        f"未识别明确缺陷的投诉数："
        f"{len(no_defect_records)}"
    )

    logger.info(
        f"处理失败投诉数："
        f"{len(failed_records)}"
    )

    logger.info(
        f"原始文本为空投诉数："
        f"{len(empty_records)}"
    )

    logger.info(
        f"最终Excel总行数："
        f"{len(final_df)}"
    )

    logger.info(
        f"结果保存位置：{output_file}"
    )

    logger.info(
        "=" * 60
    )


# =========================================================
# 12. 程序入口
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )