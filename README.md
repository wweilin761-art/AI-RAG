# AI-RAG
本地代码存档
项目结构：
rag_full_integration/
├── .env                          # 环境变量配置
├── requirements.txt              # 依赖列表
├── main.py                       # 【主入口】运行此文件
├── core/
│   ├── __init__.py
│   ├── config.py                 # 配置管理
│   ├── doc_processor.py          # 全量文档处理
│   ├── multimodal_embedding.py   # 多模态向量化
│   ├── parent_child_index.py     # 父子分层索引
│   ├── rag_retriever.py          # 多路召回 + 精排
│   ├── memory_manager.py         # 优化记忆管理
│   └── langgraph_agent.py        # LangGraph 流程编排
└── data/
    ├── uploads/
    ├── images/
    └── processed/