function pad(num) {
  return (`0${num}`).slice(-2);
}

export function hhmmss(seconds) {
  // ignore the decimal part
  const secs = Math.round(seconds);
  let minutes = Math.floor(secs / 60);
  const sec = secs % 60;
  const hours = Math.floor(minutes / 60);
  minutes %= 60;
  let str = `${pad(minutes)}:${pad(sec)}`;
  if (hours) str = pad(hours) + str;
  return str;
}

export default { hhmmss };

// bulma navbar
document.addEventListener('DOMContentLoaded', () => {

  // Get all "navbar-burger" elements
  const $navbarBurgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);

  // Check if there are any navbar burgers
  if ($navbarBurgers.length > 0) {

    // Add a click event on each of them
    $navbarBurgers.forEach(el => {
      el.addEventListener('click', () => {

        // Get the target from the "data-target" attribute
        const target = el.dataset.target;
        const $target = document.getElementById(target);

        // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
        el.classList.toggle('is-active');
        $target.classList.toggle('is-active');

      });
    });
  }

});
