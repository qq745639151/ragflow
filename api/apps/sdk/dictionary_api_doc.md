# 词典功能接口文档

## 1. 接口概述

词典功能提供了专业术语的管理和使用能力，支持通过文件上传、单个添加和批量添加等方式管理专业术语，同时提供分词测试和状态查询功能。这些接口可以帮助系统更好地识别和处理专业术语，提升分词和检索的准确性。

## 2. 接口列表

| 接口名称 | 请求方法 | 接口路径 | 功能描述 |
|---------|---------|---------|----------|
| 上传词典文件 | POST | /dictionary/upload | 上传词典文件批量添加专业术语 |
| 添加单个词条 | POST | /dictionary/add_term | 向词典添加单个术语 |
| 批量添加词条 | POST | /dictionary/batch_add_terms | 批量向词典添加多个术语 |
| 测试分词效果 | POST | /dictionary/test | 测试当前词典的分词效果 |
| 获取词典状态 | GET | /dictionary/status | 获取词典状态信息 |

## 3. 接口详细说明

### 3.1 上传词典文件

**接口路径**：`/dictionary/upload`

**请求方法**：POST

**功能描述**：上传词典文件（txt格式），批量添加专业术语到系统词典中。

**请求参数**：
- 类型：multipart/form-data
- 表单字段：
  - `file`：词典文件（必填），txt格式，每行格式为：`术语 频率 词性`

**请求示例**：
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@path/to/dictionary.txt" \
  http://localhost:8000/dictionary/upload
```

**响应格式**：
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "success": true,
    "message": "Dictionary uploaded and loaded successfully"
  }
}
```

### 3.2 添加单个词条

**接口路径**：`/dictionary/add_term`

**请求方法**：POST

**功能描述**：向词典中添加单个专业术语。

**请求参数**：
- 类型：application/json
- 字段：
  - `term`：术语名称（必填）
  - `frequency`：频率，默认值为3（可选）
  - `pos`：词性，默认值为"n"（可选）

**请求示例**：
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"term": "电力系统", "frequency": 3, "pos": "n"}' \
  http://localhost:8000/dictionary/add_term
```

**响应格式**：
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "success": true,
    "message": "Term '电力系统' added successfully"
  }
}
```

### 3.3 批量添加词条

**接口路径**：`/dictionary/batch_add_terms`

**请求方法**：POST

**功能描述**：批量向词典中添加多个专业术语。

**请求参数**：
- 类型：application/json
- 字段：
  - `terms`：术语列表（必填），包含多个术语对象
    - 每个术语对象包含：
      - `term`：术语名称（必填）
      - `frequency`：频率，默认值为3（可选）
      - `pos`：词性，默认值为"n"（可选）

**请求示例**：
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"terms": [{"term": "电力系统", "frequency": 3, "pos": "n"}, {"term": "安全稳定", "frequency": 3, "pos": "n"}]}' \
  http://localhost:8000/dictionary/batch_add_terms
```

**响应格式**：
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "success": true,
    "message": "Successfully added 2 terms",
    "added_count": 2
  }
}
```

### 3.4 测试分词效果

**接口路径**：`/dictionary/test`

**请求方法**：POST

**功能描述**：使用当前词典测试文本的分词效果。

**请求参数**：
- 类型：application/json
- 字段：
  - `text`：待分词的文本（必填）

**请求示例**：
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "电力系统的安全稳定运行"}' \
  http://localhost:8000/dictionary/test
```

**响应格式**：
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "original": "电力系统的安全稳定运行",
    "tokenized": "电力系统/的/安全/稳定/运行"
  }
}
```

### 3.5 获取词典状态

**接口路径**：`/dictionary/status`

**请求方法**：GET

**功能描述**：获取词典的当前状态信息。

**请求参数**：无

**请求示例**：
```bash
curl -X GET \
  -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8000/dictionary/status
```

**响应格式**：
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": "active",
    "info": {
      "tokenizer_type": "Tokenizer",
      "has_user_dict": true,
      "user_dict_path": "None"
    }
  }
}
```

## 4. 认证方式

所有接口都需要在请求头中添加认证信息：

```
Authorization: Bearer YOUR_API_KEY
```

其中 `YOUR_API_KEY` 是您的API密钥。

## 5. 错误响应

当请求失败时，接口会返回错误信息，示例如下：

```json
{
  "code": 401,
  "message": "Authentication error: API key is invalid!",
  "data": false
}
```

常见错误码：
- 401：认证失败，API密钥无效
- 400：请求参数错误
- 500：服务器内部错误

## 6. 词典文件格式

上传的词典文件应为txt格式，每行包含一个术语，格式为：

```
术语 频率 词性
```

例如：
```
电力系统 3 n
安全稳定 3 n
调度运行 3 n
```

- 术语：专业术语的名称
- 频率：术语的使用频率，数值越大优先级越高，默认值为3
- 词性：术语的词性，如n（名词）、v（动词）等，默认值为"n"

## 7. 注意事项

1. 上传的词典文件大小不宜过大，建议控制在10MB以内
2. 词典文件编码应为UTF-8
3. 重复的术语会被自动去重，保留首次出现的版本
4. 上传词典后，系统会自动加载新的词典配置
5. 词典修改后，新的分词效果会立即生效
6. 批量添加词条时，建议每次不超过1000个词条，以确保接口响应速度