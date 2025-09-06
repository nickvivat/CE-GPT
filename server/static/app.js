// CE RAG — Mobile-first Chat UI
// Derived and simplified from the user's original implementation with accessibility & mobile polish.

const CONFIG = { API_BASE: '/api/v1' };

class RAGChatUI {
  constructor() {
    // Elements
    this.e = {
      status: document.getElementById('status'),
      messages: document.getElementById('messages'),
      welcome: document.getElementById('welcome'),
      thread: document.getElementById('thread'),
      composer: document.getElementById('composer'),
      input: document.getElementById('input'),
      send: document.getElementById('send'),
      count: document.getElementById('count'),
      settingsBtn: document.getElementById('settingsBtn'),
      settings: document.getElementById('settingsModal'),
      closeSettings: document.getElementById('closeSettings'),
      clearChatBtn: document.getElementById('clearChatBtn'),
      newChatBtn: document.getElementById('newChatBtn'),
      newChatSidebar: document.getElementById('newChatSidebar'),
      sidebar: document.getElementById('sidebar'),
      menuBtn: document.getElementById('menuBtn'),
      history: document.getElementById('history')
    };

    // State
    this.messages = []; // {id, role:'user'|'ai'|'error', content, sources?}
    this.isStreaming = false;
    this.abortController = null;

    // Init
    this.initEvents();
    this.renderWelcome();
    this.restoreHistory();
    this.configureMarked();
    if (window.hljs) hljs.highlightAll();
  }

  initEvents() {
    // Input live
    this.e.input.addEventListener('input', () => {
      const v = this.e.input.value;
      this.e.count.textContent = v.length.toString();
      this.e.send.disabled = v.trim().length === 0 || this.isStreaming;
      this.autoResize();
    }, { passive: true });

    // Submit
    this.e.composer.addEventListener('submit', (ev) => {
      ev.preventDefault();
      if (this.e.send.disabled) return;
      const text = this.e.input.value.trim();
      if (!text) return;
      this.sendMessage(text);
    });

    // Keyboard: Enter to send (Shift+Enter newline)
    this.e.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!this.e.send.disabled) this.e.composer.requestSubmit();
      }
    });

    // Settings
    this.e.settingsBtn.addEventListener('click', () => this.openSettings());
    this.e.closeSettings.addEventListener('click', () => this.closeSettings());
    this.e.settings.addEventListener('click', (e) => {
      if (e.target === this.e.settings) this.closeSettings();
    });

    // History / new chat
    this.e.newChatBtn.addEventListener('click', () => this.clearChat());
    if (this.e.newChatSidebar) {
      this.e.newChatSidebar.addEventListener('click', () => this.clearChat());
    }

    if (this.e.menuBtn && this.e.sidebar) {
      this.e.menuBtn.addEventListener('click', () => {
        this.e.sidebar.classList.toggle('hidden');
      });
    }

    // Example prompts
    document.querySelectorAll('.example').forEach(btn => {
      btn.addEventListener('click', () => {
        const small = btn.querySelector('.text-xs')?.textContent || '';
        this.e.input.value = small;
        this.e.input.dispatchEvent(new Event('input'));
        this.e.input.focus();
      });
    });

    // Clear chat in settings
    this.e.clearChatBtn.addEventListener('click', () => {
      if (confirm('Clear all chat history?')) {
        this.clearChat();
        this.closeSettings();
      }
    });

    // Resize observer to keep to bottom
    new ResizeObserver(() => this.scrollToBottom()).observe(this.e.thread);
  }

  configureMarked() {
    if (window.marked) {
      marked.setOptions({ gfm: true, breaks: true });
    }
  }

  getSettings() {
    return {
      topK: parseInt(document.getElementById('settingsTopK').value, 10),
      language: document.getElementById('settingsLanguage').value,
      reranking: document.getElementById('settingsReranking').checked,
      includeSources: document.getElementById('settingsIncludeSources').checked
    };
  }

  autoResize() {
    const ta = this.e.input;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 140) + 'px';
  }

  setStatus(txt) { this.e.status.textContent = txt; }

  renderWelcome() {
    const hasMsgs = this.messages.length > 0;
    this.e.welcome.classList.toggle('hidden', hasMsgs);
    this.e.thread.classList.toggle('hidden', !hasMsgs);
  }

  addMessage({ role, content, sources }) {
    const id = Date.now().toString() + Math.random().toString(16).slice(2);
    const msg = { id, role, content, sources };
    this.messages.push(msg);
    this.renderMessage(msg);
    this.persist();
    this.renderWelcome();
  }

  renderMessage(m) {
    const wrap = document.createElement('div');
    wrap.className = 'mb-4';
    wrap.dataset.id = m.id;

    if (m.role === 'user') {
      wrap.innerHTML = `
        <div class="flex justify-end">
          <div class="max-w-[85%] text-white px-4 py-3 rounded-2xl border border-gray-200 dark:border-gray-700">
            <p class="text-sm whitespace-pre-wrap">${this.escape(m.content)}</p>
          </div>
        </div>`;
    } else if (m.role === 'error') {
      wrap.innerHTML = `
        <div class="flex justify-start">
          <div class="w-full max-w-3xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3 rounded-2xl">
            <p class="text-sm text-red-700 dark:text-red-300">${this.escape(m.content)}</p>
          </div>
        </div>`;
    } else {
      const md = this.renderMarkdown(m.content);
      const src = this.renderSources(m.sources);
      wrap.innerHTML = `
        <div class="flex justify-start">
          <div class="w-full max-w-3xl px-4 py-3">
            <div class="prose prose-sm max-w-none dark:prose-invert">${md}</div>
            ${src}
          </div>
        </div>`;
      wrap.querySelectorAll('pre code').forEach(block => window.hljs && hljs.highlightElement(block));
    }

    this.e.thread.appendChild(wrap);
    this.scrollToBottom();
  }

  renderMarkdown(content) {
    try {
      if (!window.marked) return `<pre class="whitespace-pre-wrap text-sm">${this.escape(content)}</pre>`;
      const html = marked.parse(content);
      return html
        .replace(/<table>/g, '<div class="overflow-x-auto my-3"><table class="min-w-full border-collapse border border-gray-300 dark:border-gray-700">')
        .replace(/<\/table>/g, '</table></div>')
        .replace(/<th>/g, '<th class="border border-gray-300 dark:border-gray-700 px-3 py-2 bg-gray-50 dark:bg-gray-800 text-left text-sm font-semibold">')
        .replace(/<td>/g, '<td class="border border-gray-300 dark:border-gray-700 px-3 py-2 text-sm">');
    } catch {
      return `<pre class="whitespace-pre-wrap text-sm">${this.escape(content)}</pre>`;
    }
  }

  renderSources(sources) {
    if (!sources || !sources.length) return '';
    const cards = sources.map((s, i) => `
      <div class="p-3 rounded-xl border border-gray-200 dark:border-gray-800 bg-white/60 dark:bg-gray-900/60">
        <div class="text-[11px] text-gray-500 mb-1">Source ${i + 1}${s.similarity_score != null ? ` • Score ${(s.similarity_score * 100).toFixed(1)}%` : ''}</div>
        <div class="text-xs text-gray-700 dark:text-gray-300">${this.escape((s.content || '').slice(0, 220))}${(s.content || '').length > 220 ? '…' : ''}</div>
      </div>`).join('');
    return `<div class="mt-3 space-y-2">${cards}</div>`;
  }

  escape(t) {
    const d = document.createElement('div');
    d.textContent = t ?? '';
    return d.innerHTML;
  }

  scrollToBottom() {
    // Smooth only if near bottom
    const el = this.e.messages;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (nearBottom) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }

  showTyping() {
    if (document.getElementById('typing')) return;
    const node = document.createElement('div');
    node.id = 'typing';
    node.className = 'mb-4 flex justify-start';
    node.innerHTML = `
      <div class="w-full max-w-3xl px-4 py-3">
        <div class="flex items-center gap-1">
          <div class="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
          <div class="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style="animation-delay:.1s"></div>
          <div class="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style="animation-delay:.2s"></div>
          <span class="ml-2 text-xs text-gray-500">Searching…</span>
        </div>
      </div>`;
    this.e.thread.appendChild(node);
    this.scrollToBottom();
  }

  hideTyping() {
    const n = document.getElementById('typing');
    if (n) n.remove();
  }

  updateStreaming(content) {
    let n = document.getElementById('streaming');
    if (!n) {
      n = document.createElement('div');
      n.id = 'streaming';
      n.className = 'mb-4 flex justify-start';
      n.innerHTML = `
        <div class="w-full max-w-3xl px-4 py-3">
          <div class="prose prose-sm max-w-none dark:prose-invert">
            <div id="streaming-content"></div>
            <span class="inline-block w-1 h-4 bg-brand-500 animate-pulse ml-1"></span>
          </div>
        </div>`;
      this.e.thread.appendChild(n);
    }
    const c = n.querySelector('#streaming-content');
    if (content) {
      c.innerHTML = this.renderMarkdown(content);
      c.querySelectorAll('pre code').forEach(block => {
        if (window.hljs) hljs.highlightElement(block);
      });
    }
    this.scrollToBottom();
  }

  removeStreaming() {
    const n = document.getElementById('streaming');
    if (n) n.remove();
  }

  async sendMessage(text) {
    this.addMessage({ role: 'user', content: text });
    this.e.input.value = '';
    this.e.input.dispatchEvent(new Event('input'));
    this.showTyping();
    this.setStatus('Searching & generating…');
    this.isStreaming = true;
    this.e.send.disabled = true;
    this.abortController = new AbortController();

    this._streamedText = '';
    this._streamSources = [];

    try {
      const settings = this.getSettings();
      const payload = {
        query: text,
        top_k: settings.topK,
        language: settings.language,
        use_reranking: settings.reranking,
        include_sources: settings.includeSources
      };

      await this.streamResponse(payload, ({ type, content, data }) => {
        if (type === 'chunk') {
          this.hideTyping();
          this.updateStreaming(this._streamedText);
        } else if (type === 'search_results') {
          this._streamSources = data || [];
        }
      });

      const finalText = this._streamedText || '';
      this.removeStreaming();
      this.hideTyping();
      this.addMessage({ role: 'ai', content: finalText, sources: this._streamSources });
    } catch (err) {
      console.error(err);
      this.hideTyping();
      this.removeStreaming();
      const msg = err?.message || 'Something went wrong.';
      this.addMessage({ role: 'error', content: `Sorry, I encountered an error: ${msg}` });
    } finally {
      this.isStreaming = false;
      this.e.send.disabled = this.e.input.value.trim().length === 0;
      this.setStatus('Ready');
      this._streamedText = '';
      this._streamSources = [];
      this.abortController = null;
    }
  }

  async streamResponse(params, onEvent) {
    const res = await fetch(`${CONFIG.API_BASE}/generate/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
      signal: this.abortController?.signal
    });
    if (!res.ok || !res.body) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const json = line.slice(5).trim();
        if (!json) continue;
        try {
          const evt = JSON.parse(json);
          if (evt.type === 'search_results') {
            onEvent({ type: 'search_results', data: evt.data });
          } else if (evt.content) {
            this._streamedText = (this._streamedText || '') + evt.content;
            onEvent({ type: 'chunk', content: evt.content });
          }
        } catch (e) {
          console.warn('Bad stream chunk', e);
        }
      }
    }
  }

  persist() {
    const compact = this.messages.slice(-20).map(({ role, content }) => ({ role, content }));
    localStorage.setItem('ce_rag_history', JSON.stringify(compact));
    this.renderHistory();
  }

  restoreHistory() {
    try {
      const raw = localStorage.getItem('ce_rag_history');
      if (!raw) return;
      const arr = JSON.parse(raw);
      arr.forEach(m => this.addMessage({ role: m.role, content: m.content }));
    } catch {}
  }

  renderHistory() {
    const h = this.e.history;
    if (!h) return; // Skip if history element doesn't exist
    h.innerHTML = '';
    const turns = this.messages.filter(m => m.role === 'user').slice(-20);
    if (turns.length === 0) {
      h.innerHTML = '<div class="text-xs text-gray-500">No history yet.</div>';
      return;
    }
    for (const t of turns.reverse()) {
      const btn = document.createElement('button');
      btn.className = 'w-full text-left px-3 py-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 text-sm';
      btn.textContent = t.content.slice(0, 64) + (t.content.length > 64 ? '…' : '');
      btn.addEventListener('click', () => {
        this.e.input.value = t.content;
        this.e.input.dispatchEvent(new Event('input'));
        this.e.input.focus();
      });
      h.appendChild(btn);
    }
  }

  clearChat() {
    this.messages = [];
    this.e.thread.innerHTML = '';
    this.persist();
    this.renderWelcome();
  }

  openSettings() {
    this.e.settings.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  }

  closeSettings() {
    this.e.settings.classList.add('hidden');
    document.body.style.overflow = 'auto';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.app = new RAGChatUI();
});
