# 008960 申万指数精确缺点复核与证书调用（2026-11-17）

本轮正常公开、只读、有界取件。未关闭 TLS 校验、未改全局证书设置。

生产补件若使用 httpx 访问申万官网，应显式使用系统证书库：

```python
import ssl
import httpx
with httpx.Client(verify=ssl.create_default_context(), trust_env=False, timeout=20.0) as client:
    response = client.get(url, params=params)
    response.raise_for_status()
```

这与 verify=False 不同：仍校验证书链和主机名。本轮缺省打包证书库报 CERTIFICATE_VERIFY_FAILED，改用上面系统信任链后，官网历史趋势 HTTPS 正常返回 HTTP 200、业务 code=200、3079 条实际记录。该原件精确缺少 2026-10-15（前后日期存在）。官网日分析端点同日 HTTP 200、实际 count=0。故当前缺项是官方回包未包含该日，不再只归因网络故障。Tushare sw_daily 权限失败、index_daily 空行另保留。

实际回执：monthly-active/sw-gap-followup/verified-truststore-attempts.json、daily-analysis-20261015-attempt.json；原始回包及摘要同目录。当前本基金合同分段计算仍严格报精确价格缺项，4个绩效维度为空，其他4维可用。不得由10月16日涨跌幅逆推15日，不补前值，不借其他指数替代。

当前 active 生产构造器消费已绑定原件，不负责网络下载。后续正常补件适配器复用上述证书方式并保存原始回包后，仍必须通过原精确日期/指数身份/合同分段门控；HTTP成功不等于所需价格齐备。
