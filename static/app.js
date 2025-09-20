const el = (q) => document.querySelector(q);
const chat = el('#chat');
const userInput = el('#userId');
const startBtn = el('#startBtn');
const msgInput = el('#message');
const sendBtn = el('#sendBtn');

let currentUser = '';

function addMessage(role, text){
  const row = document.createElement('div');
  row.className = `msg ${role==='user'?'user-msg':'assistant-msg'}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = renderText(text);
  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

function renderText(text){
  // basic formatting: bold, italics, bullets, newlines
  let t = (text||'')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,'<em>$1</em>')
    .replace(/\n\n/g,'</p><p>')
    .replace(/\n-/g,'<br>•');
  return `<p>${t}</p>`;
}

let typingEl = null;
function showTyping(){
  if(typingEl) return;
  typingEl = document.createElement('div');
  typingEl.className = 'msg assistant-msg';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const dots = document.createElement('div');
  dots.className = 'typing';
  dots.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  bubble.appendChild(dots);
  typingEl.appendChild(bubble);
  chat.appendChild(typingEl);
  chat.scrollTop = chat.scrollHeight;
}
function hideTyping(){
  if(!typingEl) return;
  // Only remove if it is still attached to chat
  if(typingEl.parentNode === chat){
    chat.removeChild(typingEl);
  }
  typingEl = null;
}

async function startSession(){
  const userId = userInput.value.trim();
  if(!userId){
    alert('Enter your name');
    return;
  }
  currentUser = userId;
  showTyping();
  try{
    const res = await fetch(`${window.API_BASE}/api/start_session`,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({user_id:userId})
    });
    const data = await res.json();
    hideTyping();
    chat.innerHTML = '';
    if(res.ok && data && data.message){
      addMessage('assistant', data.message);
    }else{
      addMessage('assistant', 'Welcome back! You can start chatting.');
      console.error('Unexpected start_session payload:', data);
    }
  }catch(err){
    hideTyping();
    addMessage('assistant', 'Failed to start session. Please try again.');
    console.error(err);
  }
}

async function send(){
  const text = msgInput.value.trim();
  if(!text || !currentUser) return;
  addMessage('user', text);
  msgInput.value = '';
  showTyping();
  try{
    const res = await fetch(`${window.API_BASE}/api/chat`,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({user_id: currentUser, message: text})
    });
    const data = await res.json();
    hideTyping();
    if(res.ok && data && data.response){
      addMessage('assistant', data.response);
    }else{
      addMessage('assistant', 'Hmm, I didn’t get a response. Please try again.');
      console.error('Unexpected chat payload:', data);
    }
  }catch(err){
    hideTyping();
    addMessage('assistant', 'Network error. Please try again.');
    console.error(err);
  }
}

startBtn.addEventListener('click', startSession);
sendBtn.addEventListener('click', send);
msgInput.addEventListener('keydown', (e)=>{
  if(e.key==='Enter') send();
});

