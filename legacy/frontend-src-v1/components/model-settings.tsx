import React, { useState, useEffect } from 'react';
import { Sliders, Cpu, Database, Network, RefreshCw, Check, AlertCircle, Wifi, WifiOff, Save, Trash2, HelpCircle } from 'lucide-react';

interface ModelConfig {
  id: string;
  providerName: string;
  baseUrl: string;
  category: string;
  modelName: string;
  isActive: number;
}

const API_BASE = 'http://localhost:8000/api/v1';

export function ModelSettings() {
  const [configs, setConfigs] = useState<ModelConfig[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  // 供应商表单数据
  const [providerName, setProviderName] = useState('SiliconFlow');
  const [baseUrl, setBaseUrl] = useState('https://api.siliconflow.cn/v1');
  const [apiKey, setApiKey] = useState('');
  
  // 自动获取的模型列表 (按用途归类)
  const [fetchedModels, setFetchedModels] = useState<Record<string, string[]>>({
    text: [],
    embedding: [],
    multimodal: [],
    rerank: []
  });
  const [isFetchingModels, setIsFetchingModels] = useState(false);
  
  // 各类别的选定绑定状态
  const [activeSelections, setActiveSelections] = useState<Record<string, { modelName: string; configId: string }>>({
    text: { modelName: '', configId: '' },
    embedding: { modelName: '', configId: '' },
    multimodal: { modelName: '', configId: '' },
    rerank: { modelName: '', configId: '' }
  });

  // 连接测试结果状态
  const [testResults, setTestResults] = useState<Record<string, { loading: boolean; success: boolean | null; latency: number; message: string }>>({
    text: { loading: false, success: null, latency: 0, message: '' },
    embedding: { loading: false, success: null, latency: 0, message: '' },
    multimodal: { loading: false, success: null, latency: 0, message: '' },
    rerank: { loading: false, success: null, latency: 0, message: '' }
  });

  // 获取所有已保存的模型配置
  const fetchConfigs = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/models/configs`);
      if (res.ok) {
        const data: ModelConfig[] = await res.json();
        setConfigs(data);
        
        // 解析当前激活的绑定
        const selections = {
          text: { modelName: '', configId: '' },
          embedding: { modelName: '', configId: '' },
          multimodal: { modelName: '', configId: '' },
          rerank: { modelName: '', configId: '' }
        };
        data.forEach(c => {
          if (c.isActive === 1) {
            selections[c.category as keyof typeof selections] = {
              modelName: c.modelName,
              configId: c.id
            };
          }
        });
        setActiveSelections(selections);
      }
    } catch (e) {
      console.error('Failed to fetch model configs', e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchConfigs();
  }, []);

  // 从服务商拉取可用模型列表
  const handleFetchModels = async () => {
    if (!baseUrl.trim()) return;
    setIsFetchingModels(true);
    try {
      const res = await fetch(`${API_BASE}/models/fetch-list`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ baseUrl, apiKey })
      });
      if (res.ok) {
        const data = await res.json();
        setFetchedModels(data.models);
        // 自动用拉取到的第一个模型填充各类选择
        const selections = { ...activeSelections };
        Object.keys(data.models).forEach(cat => {
          if (data.models[cat].length > 0 && !selections[cat].modelName) {
            selections[cat].modelName = data.models[cat][0];
          }
        });
        setActiveSelections(selections);
      }
    } catch (e) {
      console.error('Failed to fetch provider models', e);
    } finally {
      setIsFetchingModels(false);
    }
  };

  // 测试特定类别的连接
  const testConnection = async (category: string) => {
    const selection = activeSelections[category];
    if (!selection.modelName) return;

    // 找到这个模型名对应的已保存配置，如果没有保存，则使用当前输入框的临时供应商配置进行测试
    // 这提供了极佳的用户体验！
    let testBaseUrl = baseUrl;
    let testApiKey = apiKey;
    
    // 如果已经保存过，从保存数据里拉
    const savedConfig = configs.find(c => c.category === category && c.modelName === selection.modelName);
    if (savedConfig) {
      testBaseUrl = savedConfig.baseUrl;
      // 注意：apiKey 会脱敏，如果脱敏则发送空（后端会自动读原有 key 进行测试）
      testApiKey = savedConfig.id; // 我们后端在 test 接口可以通过 id 查已存的，或者在这里传
    }

    setTestResults(prev => ({
      ...prev,
      [category]: { loading: true, success: null, latency: 0, message: '' }
    }));

    try {
      const res = await fetch(`${API_BASE}/models/test-connection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          providerName: savedConfig ? savedConfig.providerName : providerName,
          baseUrl: testBaseUrl,
          apiKey: testApiKey.length > 30 ? testApiKey : apiKey, // 如果是 uuid 说明是保存过的，传空，否则传当前输入的 key
          category,
          modelName: selection.modelName
        })
      });
      if (res.ok) {
        const data = await res.json();
        setTestResults(prev => ({
          ...prev,
          [category]: {
            loading: false,
            success: data.success,
            latency: data.latency,
            message: data.message
          }
        }));
      }
    } catch (e) {
      setTestResults(prev => ({
        ...prev,
        [category]: {
          loading: false,
          success: false,
          latency: 0,
          message: `Network error: ${e}`
        }
      }));
    }
  };

  // 保存并激活某个分类的配置
  const saveAndActivateConfig = async (category: string) => {
    const selection = activeSelections[category];
    if (!selection.modelName) return;

    // 如果是基于已存配置切换，直接发激活
    const alreadySaved = configs.find(
      c => c.category === category && 
      c.modelName === selection.modelName && 
      c.providerName === providerName
    );

    try {
      let targetId = '';
      if (alreadySaved) {
        targetId = alreadySaved.id;
      } else {
        // 先创建/保存这个模型配置
        const saveRes = await fetch(`${API_BASE}/models/configs`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            providerName,
            baseUrl,
            apiKey,
            category,
            modelName: selection.modelName
          })
        });
        if (saveRes.ok) {
          const savedData = await saveRes.json();
          targetId = savedData.id;
        } else {
          alert('保存模型配置失败');
          return;
        }
      }

      // 激活该配置
      const actRes = await fetch(`${API_BASE}/models/configs/${targetId}/activate`, {
        method: 'POST'
      });
      if (actRes.ok) {
        // 重新拉取
        await fetchConfigs();
      }
    } catch (e) {
      console.error('Failed to save and activate', e);
      alert('保存并激活操作发生异常');
    }
  };

  // 删除模型配置
  const handleDeleteConfig = async (configId: string) => {
    try {
      const res = await fetch(`${API_BASE}/models/configs/${configId}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        fetchConfigs();
      }
    } catch (e) {
      console.error('Failed to delete config', e);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '20px', height: '100%', overflow: 'hidden' }} className="animate-fade-in">
      {/* 左侧：供应商配置与自动拉取 */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', fontWeight: 600 }}>
          <Sliders size={18} style={{ color: 'var(--brand-blue)' }} />
          添加供应商 / 拉取模型
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', background: 'rgba(0,0,0,0.15)', padding: '14px', borderRadius: '10px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>供应商预设:</span>
            <select 
              value={providerName} 
              onChange={(e) => {
                const name = e.target.value;
                setProviderName(name);
                if (name === 'OpenAI') {
                  setBaseUrl('https://api.openai.com/v1');
                } else if (name === 'DeepSeek') {
                  setBaseUrl('https://api.deepseek.com/v1');
                } else if (name === 'SiliconFlow') {
                  setBaseUrl('https://api.siliconflow.cn/v1');
                } else if (name === 'Ollama') {
                  setBaseUrl('http://localhost:11434/v1');
                } else if (name === 'Gemini') {
                  setBaseUrl('https://generativelanguage.googleapis.com/v1beta/openai');
                } else if (name === 'Volcengine') {
                  setBaseUrl('https://ark.cn-beijing.volces.com/api/v3');
                } else if (name === 'Alibaba') {
                  setBaseUrl('https://dashscope.aliyuncs.com/compatible-mode/v1');
                } else if (name === 'Zhipu') {
                  setBaseUrl('https://open.bigmodel.cn/api/paas/v4');
                } else if (name === 'Moonshot') {
                  setBaseUrl('https://api.moonshot.cn/v1');
                }
              }}
              style={{ width: '100%' }}
            >
              <option value="SiliconFlow">SiliconFlow (推荐/极速)</option>
              <option value="DeepSeek">DeepSeek 官方 API</option>
              <option value="OpenAI">OpenAI 官方 API</option>
              <option value="Gemini">Google Gemini API</option>
              <option value="Volcengine">火山引擎 (方舟Ark)</option>
              <option value="Alibaba">阿里百炼 (通义千问)</option>
              <option value="Zhipu">智谱 AI (GLM)</option>
              <option value="Moonshot">月之暗面 (Kimi)</option>
              <option value="Ollama">Ollama (本地部署)</option>
              <option value="Custom">自定义 (Custom)</option>
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>API Base URL:</span>
            <input 
              type="text" 
              placeholder="Base URL" 
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              style={{ width: '100%', fontSize: '12px' }}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>API Key:</span>
            <input 
              type="password" 
              placeholder="密钥" 
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              style={{ width: '100%', fontSize: '12px' }}
            />
          </div>

          <button 
            onClick={handleFetchModels}
            disabled={isFetchingModels || !baseUrl.trim()}
            className="secondary"
            style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', fontSize: '12px', padding: '8px' }}
          >
            {isFetchingModels ? (
              <>
                <RefreshCw size={12} style={{ animation: 'spin 1.5s linear infinite' }} />
                读取列表中...
              </>
            ) : (
              <>
                <RefreshCw size={12} />
                自动拉取模型列表
              </>
            )}
          </button>
        </div>

        {/* 已保存的配置清单 */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 600 }}>已注册配置 ({configs.length})</span>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', overflowY: 'auto' }}>
            {configs.map((c) => (
              <div 
                key={c.id}
                style={{ 
                  padding: '8px 10px', 
                  background: c.isActive === 1 ? 'rgba(0, 210, 255, 0.04)' : 'rgba(255, 255, 255, 0.02)',
                  border: c.isActive === 1 ? '1px solid var(--border-active)' : '1px solid var(--border-light)',
                  borderRadius: '8px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '11px'
                }}
              >
                <div style={{ overflow: 'hidden', flex: 1 }}>
                  <div style={{ display: 'flex', justifySelf: 'flex-start', gap: '4px', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{c.providerName}</span>
                    <span style={{ color: 'var(--text-muted)' }}>({c.category})</span>
                  </div>
                  <div style={{ color: 'var(--text-secondary)', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', marginTop: '2px' }} title={c.modelName}>
                    {c.modelName}
                  </div>
                </div>
                <Trash2 
                  size={12} 
                  style={{ color: 'var(--text-muted)', cursor: 'pointer' }}
                  onClick={() => {
                    if (confirm('确定删除该配置项吗？')) {
                      handleDeleteConfig(c.id);
                    }
                  }}
                />
              </div>
            ))}
            {configs.length === 0 && (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '11px', marginTop: '20px' }}>
                无保存配置
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 右侧：四大核心配置卡片与连接测试 */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto', paddingRight: '4px' }}>
        <div style={{ display: 'flex', gap: '8px', fontSize: '12px', color: 'rgba(0, 210, 255, 0.8)', background: 'rgba(0, 210, 255, 0.05)', padding: '10px 14px', borderRadius: '8px', border: '1px solid rgba(0, 210, 255, 0.1)' }}>
          <AlertCircle size={14} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong>自由模型叠加组合 (Hybrid Architecture)</strong>：您可以针对四大核心用途绑定不同的供应商模型。例如，使用 SiliconFlow 的 Qwen 模型充当文本问答以降低延迟，同时使用 OpenAI 的 Embedding 来保障精确度，不同模型类型无缝结合。
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          
          {/* Card 1: Text Generation */}
          <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h4 style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '14px', fontWeight: 600 }}>
                <Cpu size={14} style={{ color: 'var(--brand-blue)' }} />
                1. 文本大模型 (Text LLM)
              </h4>
              {activeSelections.text.configId ? (
                <span style={{ fontSize: '10px', background: 'rgba(16, 185, 129, 0.1)', color: '#10b981', padding: '2px 6px', borderRadius: '10px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                  已激活
                </span>
              ) : (
                <span style={{ fontSize: '10px', background: 'rgba(245, 158, 11, 0.1)', color: '#f59e0b', padding: '2px 6px', borderRadius: '10px' }}>
                  未配置 (Mock模式)
                </span>
              )}
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>绑定模型:</span>
                <div style={{ display: 'flex', gap: '6px' }}>
                  <select 
                    value={activeSelections.text.modelName}
                    onChange={(e) => setActiveSelections(prev => ({
                      ...prev,
                      text: { ...prev.text, modelName: e.target.value }
                    }))}
                    style={{ flex: 1, padding: '6px 10px', fontSize: '12px' }}
                  >
                    <option value="">-- 请选择 --</option>
                    {fetchedModels.text.map(m => <option key={m} value={m}>{m}</option>)}
                    {configs.filter(c => c.category === 'text').map(c => <option key={c.id} value={c.modelName}>{c.modelName} ({c.providerName})</option>)}
                  </select>
                </div>
              </div>
            </div>

            {/* 连接测试与激活 */}
            {activeSelections.text.modelName && (
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '8px', marginTop: '6px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <button 
                    onClick={() => testConnection('text')} 
                    disabled={testResults.text.loading}
                    className="secondary" 
                    style={{ padding: '4px 10px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
                  >
                    {testResults.text.loading ? <RefreshCw size={10} style={{ animation: 'spin 1s linear infinite' }} /> : <Wifi size={10} />}
                    测试连接
                  </button>
                  <button 
                    onClick={() => saveAndActivateConfig('text')} 
                    className="primary" 
                    style={{ padding: '4px 10px', fontSize: '11px', background: 'var(--brand-gradient)', display: 'flex', alignItems: 'center', gap: '4px' }}
                  >
                    <Save size={10} />
                    保存并激活
                  </button>
                </div>
                
                {testResults.text.success !== null && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', marginTop: '8px', color: testResults.text.success ? '#10b981' : '#ef4444' }}>
                    {testResults.text.success ? <Wifi size={10} /> : <WifiOff size={10} />}
                    <span>{testResults.text.success ? `握手成功! 延迟 ${testResults.text.latency}ms` : testResults.text.message}</span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Card 2: Embedding */}
          <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h4 style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '14px', fontWeight: 600 }}>
                <Database size={14} style={{ color: 'var(--color-bm25)' }} />
                2. 词嵌入模型 (Embedding)
              </h4>
              {activeSelections.embedding.configId ? (
                <span style={{ fontSize: '10px', background: 'rgba(16, 185, 129, 0.1)', color: '#10b981', padding: '2px 6px', borderRadius: '10px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                  已激活
                </span>
              ) : (
                <span style={{ fontSize: '10px', background: 'rgba(245, 158, 11, 0.1)', color: '#f59e0b', padding: '2px 6px', borderRadius: '10px' }}>
                  未配置 (Mock模式)
                </span>
              )}
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>绑定模型:</span>
                <select 
                  value={activeSelections.embedding.modelName}
                  onChange={(e) => setActiveSelections(prev => ({
                    ...prev,
                    embedding: { ...prev.embedding, modelName: e.target.value }
                  }))}
                  style={{ width: '100%', padding: '6px 10px', fontSize: '12px' }}
                >
                  <option value="">-- 请选择 --</option>
                  {fetchedModels.embedding.map(m => <option key={m} value={m}>{m}</option>)}
                  {configs.filter(c => c.category === 'embedding').map(c => <option key={c.id} value={c.modelName}>{c.modelName} ({c.providerName})</option>)}
                </select>
              </div>
            </div>

            {activeSelections.embedding.modelName && (
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '8px', marginTop: '6px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <button 
                    onClick={() => testConnection('embedding')} 
                    disabled={testResults.embedding.loading}
                    className="secondary" 
                    style={{ padding: '4px 10px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
                  >
                    {testResults.embedding.loading ? <RefreshCw size={10} style={{ animation: 'spin 1s linear infinite' }} /> : <Wifi size={10} />}
                    测试连接
                  </button>
                  <button 
                    onClick={() => saveAndActivateConfig('embedding')} 
                    className="primary" 
                    style={{ padding: '4px 10px', fontSize: '11px', background: 'var(--brand-gradient)', display: 'flex', alignItems: 'center', gap: '4px' }}
                  >
                    <Save size={10} />
                    保存并激活
                  </button>
                </div>
                
                {testResults.embedding.success !== null && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', marginTop: '8px', color: testResults.embedding.success ? '#10b981' : '#ef4444' }}>
                    {testResults.embedding.success ? <Wifi size={10} /> : <WifiOff size={10} />}
                    <span>{testResults.embedding.success ? `握手成功! 延迟 ${testResults.embedding.latency}ms` : testResults.embedding.message}</span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Card 3: Multimodal */}
          <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h4 style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '14px', fontWeight: 600 }}>
                <Cpu size={14} style={{ color: 'var(--color-hyde)' }} />
                3. 多模态大模型 (Multimodal)
              </h4>
              {activeSelections.multimodal.configId ? (
                <span style={{ fontSize: '10px', background: 'rgba(16, 185, 129, 0.1)', color: '#10b981', padding: '2px 6px', borderRadius: '10px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                  已激活
                </span>
              ) : (
                <span style={{ fontSize: '10px', background: 'rgba(245, 158, 11, 0.1)', color: '#f59e0b', padding: '2px 6px', borderRadius: '10px' }}>
                  未配置
                </span>
              )}
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>绑定模型:</span>
                <select 
                  value={activeSelections.multimodal.modelName}
                  onChange={(e) => setActiveSelections(prev => ({
                    ...prev,
                    multimodal: { ...prev.multimodal, modelName: e.target.value }
                  }))}
                  style={{ width: '100%', padding: '6px 10px', fontSize: '12px' }}
                >
                  <option value="">-- 请选择 --</option>
                  {fetchedModels.multimodal.map(m => <option key={m} value={m}>{m}</option>)}
                  {configs.filter(c => c.category === 'multimodal').map(c => <option key={c.id} value={c.modelName}>{c.modelName} ({c.providerName})</option>)}
                </select>
              </div>
            </div>

            {activeSelections.multimodal.modelName && (
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '8px', marginTop: '6px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <button 
                    onClick={() => testConnection('multimodal')} 
                    disabled={testResults.multimodal.loading}
                    className="secondary" 
                    style={{ padding: '4px 10px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
                  >
                    {testResults.multimodal.loading ? <RefreshCw size={10} style={{ animation: 'spin 1s linear infinite' }} /> : <Wifi size={10} />}
                    测试连接
                  </button>
                  <button 
                    onClick={() => saveAndActivateConfig('multimodal')} 
                    className="primary" 
                    style={{ padding: '4px 10px', fontSize: '11px', background: 'var(--brand-gradient)', display: 'flex', alignItems: 'center', gap: '4px' }}
                  >
                    <Save size={10} />
                    保存并激活
                  </button>
                </div>
                
                {testResults.multimodal.success !== null && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', marginTop: '8px', color: testResults.multimodal.success ? '#10b981' : '#ef4444' }}>
                    {testResults.multimodal.success ? <Wifi size={10} /> : <WifiOff size={10} />}
                    <span>{testResults.multimodal.success ? `握手成功! 延迟 ${testResults.multimodal.latency}ms` : testResults.multimodal.message}</span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Card 4: Rerank */}
          <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h4 style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '14px', fontWeight: 600 }}>
                <Network size={14} style={{ color: 'var(--color-weaved)' }} />
                4. 重排模型 (Rerank)
              </h4>
              {activeSelections.rerank.configId ? (
                <span style={{ fontSize: '10px', background: 'rgba(16, 185, 129, 0.1)', color: '#10b981', padding: '2px 6px', borderRadius: '10px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                  已激活
                </span>
              ) : (
                <span style={{ fontSize: '10px', background: 'rgba(245, 158, 11, 0.1)', color: '#f59e0b', padding: '2px 6px', borderRadius: '10px' }}>
                  未配置 (大模型评分)
                </span>
              )}
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>绑定模型:</span>
                <select 
                  value={activeSelections.rerank.modelName}
                  onChange={(e) => setActiveSelections(prev => ({
                    ...prev,
                    rerank: { ...prev.rerank, modelName: e.target.value }
                  }))}
                  style={{ width: '100%', padding: '6px 10px', fontSize: '12px' }}
                >
                  <option value="">-- 请选择 --</option>
                  {fetchedModels.rerank.map(m => <option key={m} value={m}>{m}</option>)}
                  {configs.filter(c => c.category === 'rerank').map(c => <option key={c.id} value={c.modelName}>{c.modelName} ({c.providerName})</option>)}
                </select>
              </div>
            </div>

            {activeSelections.rerank.modelName && (
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '8px', marginTop: '6px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <button 
                    onClick={() => testConnection('rerank')} 
                    disabled={testResults.rerank.loading}
                    className="secondary" 
                    style={{ padding: '4px 10px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
                  >
                    {testResults.rerank.loading ? <RefreshCw size={10} style={{ animation: 'spin 1s linear infinite' }} /> : <Wifi size={10} />}
                    测试连接
                  </button>
                  <button 
                    onClick={() => saveAndActivateConfig('rerank')} 
                    className="primary" 
                    style={{ padding: '4px 10px', fontSize: '11px', background: 'var(--brand-gradient)', display: 'flex', alignItems: 'center', gap: '4px' }}
                  >
                    <Save size={10} />
                    保存并激活
                  </button>
                </div>
                
                {testResults.rerank.success !== null && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', marginTop: '8px', color: testResults.rerank.success ? '#10b981' : '#ef4444' }}>
                    {testResults.rerank.success ? <Wifi size={10} /> : <WifiOff size={10} />}
                    <span>{testResults.rerank.success ? `握手成功! 延迟 ${testResults.rerank.latency}ms` : testResults.rerank.message}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
