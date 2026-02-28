# VALIDATION_REPORT_v8

## 0) 前提与输入证据
已读取并作为本轮起点：
- `VALIDATION_REPORT_v2.md`
- `VALIDATION_REPORT_v4.md`
- `VALIDATION_REPORT_v5.md`
- `VALIDATION_REPORT_v6.md`
- `VALIDATION_REPORT_v7.md`
- `openai_sdk_gate_v7.log`
- `crew_kickoff_v7.log`
- `route_baseline_v7.log`

v7 起点结论复核：
- `gpt-5.2-codex` 路由正确并可通过。
- `gpt-5.2-pro` 未发生 alias，但在 Platform credential 解析阶段失败。
- 关键阻塞：`refresh_token_reused`。

## 1) 本轮执行摘要（v8）
### Step 1: auth cache 脱敏探测
- 日志：`auth_cache_probe_v8.log`
- 结果：
  - `OPENAI_API_KEY` 字段存在但为空。
  - `tokens.access_token` 存在且未过期。
  - `tokens.id_token` 存在但已过期。
  - `tokens.refresh_token` 存在。
  - `id_token/access_token` 的 `organization_id` 与 `project_id` 均不存在。

### Step 2: refresh_token_reused 处理（重读+对比+加锁）
- 日志：`auth_refresh_v8.log`
- 结果：
  - 在文件锁下重读 `auth.json`，对比内存快照与磁盘值：无更新。
  - 执行一次标准 refresh：`401`, `refresh_token_reused`。
  - 判定 `needs_relogin=True`。

### Step 3: relogin 路径
- 日志：`auth_relogin_v8.log`
- 实际执行与约束：
  - 曾触发一次 `codex logout` 后进入 device-auth，随后按用户要求中断并从备份恢复 `auth.json`。
  - 用户进一步明确约束：不执行重登录（云端环境无法浏览器授权）。
  - 因此本轮不再继续 relogin。

### Step 4: Platform credential 解析
- 日志：`platform_credential_resolve_v8.log`
- 结果：
  - 优先级 1：`auth.json.OPENAI_API_KEY` 为空。
  - `id_token` 过期，refresh 再次 `refresh_token_reused`。
  - token-exchange 使用现有过期 `id_token`：`401 invalid_id_token` (`Invalid ID token: token expired`)。
  - 本轮未拿到可用 Platform key。

### Step 5: `gpt-5.2-pro` Platform 验证
- SDK 日志：`openai_sdk_gate_v8.log`
- Crew 日志：`crew_kickoff_v8.log`
- 结果：
  - `requested_model == effective_model == gpt-5.2-pro`（无 alias）
  - 路由目标保持 `https://api.openai.com/v1`
  - 但无可用 Platform credential，故失败分类为 `LOCAL_AUTH_STATE_BROKEN`

### Step 6: `gpt-5.2-codex` 回归
- SDK：PASS（Responses + `https://chatgpt.com/backend-api/codex`）
- Crew kickoff：PASS（Responses + `https://chatgpt.com/backend-api/codex`）
- credential_source：`codex_auth_json_oauth`

## 2) 官方检索结论（已执行）
### OpenAI 官方文档
1. Codex 认证文档明确两种登录方式（ChatGPT / API key），并说明登录缓存位于 `~/.codex/auth.json`（或 keyring），ChatGPT 会自动刷新 token：
- https://developers.openai.com/codex/auth/

2. 官方工程文章明确端点分流：
- ChatGPT 登录：`https://chatgpt.com/backend-api/codex/responses`
- API key 登录（OpenAI hosted）：`https://api.openai.com/v1/responses`
- https://openai.com/index/unrolling-the-codex-agent-loop/

3. 官方文章还明确提到：可通过 ChatGPT 登录并选择 API organization，让 Codex 自动生成并配置 API key：
- https://openai.com/index/introducing-codex/

### openai/codex 官方源码
1. 默认路由分流（ChatGPT auth -> chatgpt backend）
- `codex-rs/core/src/model_provider_info.rs`（`default_base_url` 逻辑）

2. 登录服务中 token-exchange 生成 API key 的实现
- `codex-rs/login/src/server.rs`（`obtain_api_key`，`requested_token=openai-api-key`）

3. refresh 失败码语义（含 `refresh_token_reused`）
- `codex-rs/core/src/auth.rs`（refresh 失败分类）

## 3) 最终用例矩阵（v8）
- `sdk.responses.gpt-5.2-codex`: PASS
- `crew.responses.gpt-5.2-codex`: PASS
- `sdk.responses.gpt-5.2-pro`: FAIL（Platform credential unavailable）
- `crew.responses.gpt-5.2-pro`: FAIL（Platform credential unavailable）

## 4) 单选结论
**LOCAL_AUTH_STATE_BROKEN**

判定依据：
1. 路由已正确拆分且稳定（codex 路由可通过，pro 路由未 alias）。
2. 本机 auth cache 无可用 `OPENAI_API_KEY`。
3. `id_token` 过期且 refresh 稳定返回 `refresh_token_reused`。
4. 在用户禁止重登录（云端无浏览器授权）的约束下，无法恢复 fresh `id_token`，从而无法派生 Platform credential。

该结论不等价于 `MODEL_UNAVAILABLE`，也不等价于 `QUOTA_OR_BILLING_BLOCKED`；本轮阻塞发生在 credential derivation 之前。
