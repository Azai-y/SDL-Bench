# =========================
# Model Configuration
# =========================

MODEL_CONFIG = {

    # 当前激活模型（统一入口）
    "active_model": "deepseek",

    # =========================
    # DeepSeek V4 Pro
    # =========================
    "deepseek": {
        "model": "deepseek-v4-pro",
        "provider": "openai_compatible",
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.2
    },

    # =========================
    # MiMo V2.5 Pro (小米)
    # =========================
    "mimo": {
        "model": "mimo-v2.5-pro",
        "provider": "openai_compatible",
        "api_url": "https://token-plan-cn.xiaomimimo.com/v1/chat/completions",
        "api_key_env": "MIMO_API_KEY",
        "temperature": 0.2,
        "max_tokens": 4096
    },

    # =========================
    # Qwen (DashScope)
    # =========================
    "qwen": {
        "model": "qwen3.7-plus",
        "provider": "dashscope",
        "api_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        "api_key_env": "DASHSCOPE_API_KEY",
        "temperature": 0.2
    },

    # =========================
    # OpenAI
    # =========================
    "openai": {
        "model": "gpt-4o-mini",
        "provider": "openai_compatible",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.2
    },

    # =========================
    # Claude
    # =========================
    "claude": {
        "model": "claude-3-opus-20240229",
        "provider": "openai_compatible",
        "api_url": "https://api.anthropic.com/v1/messages",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.2
    },

}
