// Song difficulty list (paper Table 2: steps at 20 Hz, NPS, F1).
const songs = [
  ['Gymnopédie No. 1', 1.77, 0.88],
  ['Twinkle Twinkle', 1.51, 0.86],
  ['Clair de Lune', 2.65, 0.63],
  ['French Suite No. 1, Allemande', 10.89, 0.59],
  ['French Suite No. 5, Sarabande', 3.56, 0.59],
  ['Piano Sonata No. 23, 2nd mov.', 12.80, 0.49],
  ['Pictures at an Exhibition: Great Kiev', 8.39, 0.45],
  ['Kreisleriana Op. 16 No. 8', 9.78, 0.44],
  ['Piano Sonata No. 2, 1st mov.', 6.94, 0.44],
  ['Piano Sonata K279, 1st mov.', 9.54, 0.44],
  ['Für Elise', 10.44, 0.42],
  ['French Suite No. 5, Gavotte', 6.80, 0.40],
  ['Piano Sonata D845, 1st mov.', 9.04, 0.34],
  ['Winter Wind', 16.64, 0.28],
  ['Bagatelle Op. 3 No. 4', 9.24, 0.26],
  ['Partita No. 2, 6th mov.', 11.91, 0.25],
];
const songList = document.getElementById('song-list');
if (songList) {
  songList.innerHTML = songs.map(([name, nps, f1]) => `
    <li title="${name}: NPS ${nps.toFixed(2)}, F1 ${f1.toFixed(2)}">
      <span class="name">${name} <small>· ${nps.toFixed(1)} NPS</small></span>
      <span class="bar"><i style="width:${f1 * 100}%"></i></span>
      <span class="val">${f1.toFixed(2)}</span>
    </li>`).join('');
}

// Sticky nav: show wordmark after scrolling, highlight the current section.
const nav = document.querySelector('.topnav');
const navLinks = [...document.querySelectorAll('.topnav-links a')];
const sections = navLinks.map((a) => document.querySelector(a.getAttribute('href'))).filter(Boolean);
const onScroll = () => {
  nav.classList.toggle('scrolled', window.scrollY > 240);
  let current = null;
  sections.forEach((s) => { if (s.getBoundingClientRect().top < 120) current = s; });
  navLinks.forEach((a) => a.classList.toggle('active', current && a.getAttribute('href') === `#${current.id}`));
};
window.addEventListener('scroll', onScroll, { passive: true });
onScroll();

// Video filters.
const chips = document.querySelectorAll('.chip');
const cards = document.querySelectorAll('.video-card');
chips.forEach((chip) => chip.addEventListener('click', () => {
  const f = chip.dataset.filter;
  chips.forEach((c) => { c.classList.toggle('is-active', c === chip); c.setAttribute('aria-pressed', c === chip); });
  cards.forEach((card) => { card.hidden = f !== 'all' && card.dataset.track !== f; });
}));

// Static leaderboard filters.
const leaderboardSong = document.getElementById('leaderboard-song');
const leaderboardRows = [...document.querySelectorAll('#leaderboard-body tr')];
const leaderboardTypes = [...document.querySelectorAll('.method-filters input')];
const leaderboardEmpty = document.getElementById('leaderboard-empty');
const leaderboardSortButtons = [...document.querySelectorAll('.sort-button')];
if (leaderboardSong) {
  let sortKey = 'avg-reward';
  let sortDirection = 'desc';
  const sortValue = (row, key) => {
    if (key === 'type') return row.dataset.type;
    const cell = row.querySelectorAll('td')[{ 'avg-reward': 2, tokens: 3 }[key]].textContent.trim();
    if (cell === '—') return null;
    const value = Number(cell.replaceAll(',', '').replace('K', ''));
    return cell.endsWith('K') ? value * 1000 : value;
  };
  const filterLeaderboard = () => {
    const types = new Set(leaderboardTypes.filter((input) => input.checked).map((input) => input.value));
    let rank = 0;
    leaderboardRows.forEach((row) => {
      const visible = row.dataset.song === leaderboardSong.value && types.has(row.dataset.type);
      row.hidden = !visible;
      row.classList.toggle('leader', visible && rank === 0);
      if (visible) row.querySelector('.rank').textContent = ++rank;
    });
    leaderboardEmpty.hidden = rank > 0;
  };
  const sortLeaderboard = () => {
    leaderboardRows.sort((a, b) => {
      const aValue = sortValue(a, sortKey);
      const bValue = sortValue(b, sortKey);
      if (aValue === null) return bValue === null ? 0 : 1;
      if (bValue === null) return -1;
      const comparison = typeof aValue === 'string' ? aValue.localeCompare(bValue) : aValue - bValue;
      return sortDirection === 'asc' ? comparison : -comparison;
    });
    leaderboardRows.forEach((row) => row.parentNode.appendChild(row));
    leaderboardSortButtons.forEach((button) => {
      const active = button.dataset.sort === sortKey;
      button.classList.toggle('is-active', active);
      button.parentElement.setAttribute('aria-sort', active ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none');
      button.querySelector('span').textContent = active ? (sortDirection === 'asc' ? '↑' : '↓') : '';
    });
    filterLeaderboard();
  };
  leaderboardSong.addEventListener('change', filterLeaderboard);
  leaderboardTypes.forEach((input) => input.addEventListener('change', filterLeaderboard));
  leaderboardSortButtons.forEach((button) => button.addEventListener('click', () => {
    const nextKey = button.dataset.sort;
    sortDirection = nextKey === sortKey ? (sortDirection === 'asc' ? 'desc' : 'asc') : (nextKey === 'type' ? 'asc' : 'desc');
    sortKey = nextKey;
    sortLeaderboard();
  }));
  sortLeaderboard();
}

// Figure lightbox.
const lightbox = document.getElementById('lightbox');
const lightboxImg = lightbox.querySelector('img');
document.querySelectorAll('.zoomable img').forEach((img) => img.addEventListener('click', () => {
  lightboxImg.src = img.currentSrc || img.src;
  lightboxImg.alt = img.alt;
  lightbox.hidden = false;
  document.body.style.overflow = 'hidden';
}));
const closeLightbox = () => { lightbox.hidden = true; document.body.style.overflow = ''; };
lightbox.addEventListener('click', closeLightbox);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !lightbox.hidden) closeLightbox(); });

// Copy command snippets.
document.querySelectorAll('[data-copy]').forEach((btn) => btn.addEventListener('click', async () => {
  const text = document.querySelector(btn.dataset.copy).textContent;
  try { await navigator.clipboard.writeText(text); } catch { return; }
  btn.classList.add('copied');
  btn.title = 'Copied!';
  setTimeout(() => { btn.classList.remove('copied'); btn.title = 'Copy'; }, 1600);
}));

// Featured video: play with sound. Browsers block unmuted autoplay until the
// visitor interacts, so fall back to muted playback; clicking the video toggles
// sound, and the first click anywhere else on the page also unmutes it.
const featured = document.getElementById('featured-video');
if (featured) {
  const frame = featured.closest('.featured-frame');
  const toggle = frame.querySelector('.sound-toggle');
  const events = ['pointerdown', 'keydown', 'touchstart'];
  const sync = () => {
    frame.classList.toggle('is-muted', featured.muted);
    toggle.setAttribute('aria-label', featured.muted ? 'Unmute video' : 'Mute video');
  };
  const setMuted = (muted) => {
    featured.muted = muted;
    if (featured.paused) featured.play().catch(() => {});
    sync();
  };
  const stopAutoUnmute = () => events.forEach((e) => document.removeEventListener(e, autoUnmute, true));
  function autoUnmute(e) {
    if (frame.contains(e.target)) return;
    stopAutoUnmute();
    setMuted(false);
  }
  frame.addEventListener('click', () => { stopAutoUnmute(); setMuted(!featured.muted); });
  featured.muted = false;
  featured.play().then(sync).catch(() => {
    featured.muted = true;
    featured.play().catch(() => {});
    sync();
    events.forEach((e) => document.addEventListener(e, autoUnmute, true));
  });
}
