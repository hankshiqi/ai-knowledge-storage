# 文本切块最大长度：单个文本块最多包含 1000 字符（防止过长导致向量失真）
CHUNK_MAX_SIZE = 1000
# 文本切块基准长度：单个文本块理想大小为 600 字符（兼顾语义完整性 + 检索精度）
CHUNK_SIZE = 600
# 文本块重叠长度：相邻块之间重叠 20 字符，保证语义不被切断、上下文连贯
CHUNK_OVERLAP = 50
#文本块最小长度
CHUNK_MIN_SIZE=400

#声明常量，切块的截取数量，默认:10
CHUNKS_SPLIT_TOP_NUMBER = 10
#chunks处理的batch_size
CHUNKS_BATCH_SIZE = 6