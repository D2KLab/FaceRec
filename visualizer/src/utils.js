function pad(num) {
  return (`0${num}`).slice(-2);
}

export function hhmmss(seconds) {
  // ignore the decimal part
  const secs = Math.round(seconds)
  let minutes = Math.floor(secs / 60);
  const sec = secs % 60;
  const hours = Math.floor(minutes / 60);
  minutes %= 60;
  let str = `${pad(minutes)}:${pad(sec)}`;
  if (hours) str = pad(hours) + str;
  return str;
}

export default { hhmmss };
