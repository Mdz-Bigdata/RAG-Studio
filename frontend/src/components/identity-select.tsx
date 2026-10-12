// 当前身份切换器：本项目没有登录体系，检索 / 问答时带上这里选的"级别 + 部门"做权限元数据过滤；
// "用户"区分长期记忆与个性化推荐（LanceDB），空 = default。
import { ShieldCheck, UserRound } from 'lucide-react';
import { ACCESS_LEVELS, useApp } from '../store/app-store';

export function IdentitySelect() {
  const { identity, setIdentity } = useApp();
  return (
    <span className="row gap-6" title="检索与问答按这个身份过滤文档（权限级别 + 部门）">
      <ShieldCheck size={13} className="faint" />
      <select value={identity.level} onChange={(e) => setIdentity({ ...identity, level: Number(e.target.value) })} style={{ width: 'auto', padding: '4px 26px 4px 8px', fontSize: 12 }} aria-label="当前身份级别">
        {ACCESS_LEVELS.map((l) => <option key={l.value} value={l.value}>{l.user}</option>)}
      </select>
      <input value={identity.department ?? ''} onChange={(e) => setIdentity({ ...identity, department: e.target.value || null })} placeholder="部门（可空）" style={{ width: 100, padding: '4px 8px', fontSize: 12 }} aria-label="当前身份部门" />
      <UserRound size={13} className="faint" />
      <input value={identity.userId ?? ''} onChange={(e) => setIdentity({ ...identity, userId: e.target.value.trim() || null })} placeholder="用户 default" title="长期记忆与个性化推荐按用户区分" style={{ width: 92, padding: '4px 8px', fontSize: 12 }} aria-label="当前用户" />
    </span>
  );
}
