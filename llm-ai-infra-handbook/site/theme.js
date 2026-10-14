(function () {
  function label() {
    return document.documentElement.dataset.theme === 'dark' ? '浅色' : '深色';
  }

  function mountToggle() {
    var nav = document.querySelector('.app-nav');
    if (!nav || nav.querySelector('.theme-toggle')) return;
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'theme-toggle';
    button.setAttribute('aria-label', '切换页面主题');
    button.textContent = label();
    button.addEventListener('click', function () {
      var root = document.documentElement;
      var next = root.dataset.theme === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      localStorage.setItem('handbook-theme', next);
      button.textContent = label();
    });
    nav.appendChild(button);
  }

  window.mountHandbookThemeToggle = mountToggle;
  document.addEventListener('DOMContentLoaded', mountToggle);
  window.addEventListener('hashchange', function () { setTimeout(mountToggle, 0); });
}());
