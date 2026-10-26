from typing import Any
from pydantic import BaseModel,Field

class QueryRequest(BaseModel):
    """
    查询请求参数
    """
    session_id: str = Field(None, description="会话ID")
    query: str | None = Field(..., description="原始查询")
    is_stream: bool = Field(False, description="是否流式返回")


# 错误示范
# class A:
#     lst = []
#
# a1 = A()
# a2 = A()
# a1.lst.append(1)
#
# print(a2.lst)  # 输出 [1] ！！！ 被污染了

class AsyncQueryResponse(BaseModel):
    """
    查询响应参数
    """
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="响应信息")



class QueryResponse(BaseModel):
    """
    查询响应参数
    这个类定义了一个查询响应的数据模型，继承自BaseModel，用于封装查询操作后的响应数据。
    包含会话ID、响应信息、答案、图片URL列表和已完成任务列表等字段。
    """
    session_id: str = Field(..., description="会话ID")  # 用户的会话标识符，用于区分不同的查询会话
    message: str = Field(..., description="响应信息")  # 系统返回的响应信息，通常用于描述查询状态或结果
    answer: str = Field("", description="答案")  # 查询的答案内容，默认为空字符串
    image_urls: list[str] = Field(description="图片URL列表" ,default_factory=list)  # 相关图片的URL列表，默认为空列表
    done_list: list[str]  = Field(description="已完成任务列表", default_factory=list)  # 已完成的任务标识列表，默认为空列表


class ClearHistoryResponse(BaseModel):
    """
    清空聊天记录接口响应体
    """
    message: str = Field(..., description="操作提示信息")
    deleted_count: int = Field(..., description="成功删除的消息条数")

#用于创建
class HistoryItem(BaseModel):   
    id: str = Field(default="", description="消息ID")
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    #注意不能用 list【str】= []的方式，否则会被污染，因为创建的是针对这个类的列表  有些像java当中的static对象，这个类是共享的
    #是一个“生成器/工厂”。框架（如 Pydantic 或 Dataclasses）在每次实例化对象时，只要发现你没传这个参数，它就会去调用一次 list()，现场给你造一个全新的空列表。
    #field可以夹加上元数据限制
    item_names: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    #时间戳
    ts: Any = None

class HistoryResponse(BaseModel):
    session_id: str
    items: list[HistoryItem] = Field(description="历史记录列表",default_factory=list)