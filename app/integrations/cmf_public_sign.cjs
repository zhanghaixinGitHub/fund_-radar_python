// 仅解析公开SSR配置、重放公开业务客户端；不运行校验脚本或读取登录令牌。
const fs = require('fs'), crypto = require('crypto'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const secret = input.client.match(/\$s\(n,"([a-f0-9]{32})"\)/)?.[1];
const outer = input.client.match(/const Ns=\(\)=>"([a-f0-9]{32})"/)?.[1];
const merchant = input.client.match(/this\.merchantKey=s\.merchantKey\|\|"([a-z]+)"/)?.[1];
if (!secret || !outer || !merchant || !input.client.includes('Dt.sm4.encrypt'))
  throw Error('PUBLIC_CLIENT_FORMAT_CHANGED');
const script = input.page.match(/<script>\s*(window\.__NUXT__=[\s\S]*?)<\/script>/)?.[1];
if (!script) throw Error('PUBLIC_SSR_DATA_REQUIRED');
const context = vm.createContext({window: Object.create(null)}, {codeGeneration: {strings: false, wasm: false}});
vm.runInContext(script, context, {timeout: 1000});
const conf = context.window.__NUXT__.config.public;
if (typeof conf.siteno !== 'string' || !['string', 'number'].includes(typeof conf.merchantId))
  throw Error('PUBLIC_CONFIG_SCHEMA_CHANGED');
const data = {...input.params, siteno: conf.siteno, merchantId: conf.merchantId, request_id: crypto.randomUUID()};
const key = crypto.createHash('sm3').update(secret).digest('hex').substring(0, 16);
const cipher = crypto.createCipheriv('sm4-ecb', Buffer.from(key), null);
const encrypted = Buffer.concat([cipher.update(JSON.stringify(data)), cipher.final()]).toString('base64');
const signature = crypto.createHash('sm3').update(`secret=${secret}data=${encrypted}request_id=${outer}encrypted=true`).digest('hex');
process.stdout.write(JSON.stringify({headers: {'tk-trans-merchant-key': merchant, 'tk-trans-signature': signature},
  params: {data: encrypted, request_id: outer, encrypted: 'true'}}));
