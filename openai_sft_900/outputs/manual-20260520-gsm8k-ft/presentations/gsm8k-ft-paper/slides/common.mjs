export const STYLE = {
  bg: "#F7F2E8",
  paper: "#FCFAF6",
  ink: "#112033",
  soft: "#5F6D7B",
  accent: "#1F6B7A",
  accent2: "#C57A31",
  accent3: "#DCE4E8",
  dark: "#0E1828",
  sans: "Avenir Next",
  serif: "Georgia",
};

export function bg(slide, ctx) {
  ctx.addShape(slide, {
    left: 0,
    top: 0,
    width: ctx.W,
    height: ctx.H,
    fill: STYLE.bg,
    line: ctx.line(),
  });
  ctx.addShape(slide, {
    left: 54,
    top: 44,
    width: ctx.W - 108,
    height: ctx.H - 88,
    fill: STYLE.paper,
    line: ctx.line("#E7DED1", 1),
  });
}

export function kicker(slide, ctx, text) {
  ctx.addText(slide, {
    text,
    left: 84,
    top: 72,
    width: 380,
    height: 24,
    fontSize: 10,
    bold: true,
    color: STYLE.accent,
    typeface: STYLE.sans,
  });
}

export function title(slide, ctx, text, options = {}) {
  ctx.addText(slide, {
    text,
    left: options.left ?? 84,
    top: options.top ?? 108,
    width: options.width ?? 860,
    height: options.height ?? 88,
    fontSize: options.size ?? 30,
    bold: true,
    color: STYLE.ink,
    typeface: STYLE.serif,
  });
}

export function body(slide, ctx, text, options = {}) {
  ctx.addText(slide, {
    text,
    left: options.left ?? 84,
    top: options.top ?? 200,
    width: options.width ?? 420,
    height: options.height ?? 160,
    fontSize: options.size ?? 16,
    color: options.color ?? STYLE.ink,
    typeface: options.face ?? STYLE.sans,
  });
}

export function bulletList(slide, ctx, items, options = {}) {
  const left = options.left ?? 84;
  const top = options.top ?? 220;
  const width = options.width ?? 460;
  const rowGap = options.rowGap ?? 54;

  items.forEach((item, index) => {
    const y = top + index * rowGap;
    ctx.addShape(slide, {
      left,
      top: y + 8,
      width: 8,
      height: 8,
      fill: index % 2 === 0 ? STYLE.accent : STYLE.accent2,
      line: ctx.line(),
      geometry: "ellipse",
    });
    ctx.addText(slide, {
      text: item,
      left: left + 20,
      top: y,
      width,
      height: rowGap - 4,
      fontSize: options.size ?? 15,
      color: options.color ?? STYLE.ink,
      typeface: options.face ?? STYLE.sans,
    });
  });
}

export function card(slide, ctx, options) {
  const fill = options.fill ?? "#F1ECE2";
  ctx.addShape(slide, {
    left: options.left,
    top: options.top,
    width: options.width,
    height: options.height,
    fill,
    line: ctx.line("#D8D0C4", 1),
  });
}

export function footer(slide, ctx, number, note) {
  ctx.addShape(slide, {
    left: 84,
    top: 654,
    width: ctx.W - 168,
    height: 1,
    fill: STYLE.accent3,
    line: ctx.line(),
  });
  ctx.addText(slide, {
    text: note,
    left: 84,
    top: 666,
    width: 940,
    height: 18,
    fontSize: 9.5,
    color: STYLE.soft,
    typeface: STYLE.sans,
  });
  ctx.addText(slide, {
    text: String(number).padStart(2, "0"),
    left: 1144,
    top: 662,
    width: 52,
    height: 20,
    fontSize: 10,
    bold: true,
    color: STYLE.soft,
    typeface: STYLE.sans,
    align: "right",
  });
}

export function metric(slide, ctx, options) {
  ctx.addText(slide, {
    text: options.value,
    left: options.left,
    top: options.top,
    width: options.width ?? 120,
    height: 34,
    fontSize: options.valueSize ?? 26,
    bold: true,
    color: options.valueColor ?? STYLE.ink,
    typeface: options.valueFace ?? STYLE.serif,
  });
  ctx.addText(slide, {
    text: options.label,
    left: options.left,
    top: options.top + 38,
    width: options.width ?? 140,
    height: 18,
    fontSize: options.labelSize ?? 10,
    bold: true,
    color: STYLE.soft,
    typeface: STYLE.sans,
  });
  if (options.note) {
    ctx.addText(slide, {
      text: options.note,
      left: options.left,
      top: options.top + 56,
      width: options.width ?? 160,
      height: 28,
      fontSize: 9.5,
      color: STYLE.soft,
      typeface: STYLE.sans,
    });
  }
}

export function barChart(slide, ctx, data, options = {}) {
  const left = options.left ?? 90;
  const top = options.top ?? 210;
  const width = options.width ?? 760;
  const height = options.height ?? 310;
  const maxValue = options.maxValue ?? 100;
  const barWidth = Math.floor(width / (data.length * 1.55));
  const gap = Math.floor((width - data.length * barWidth) / (data.length - 1));

  ctx.addShape(slide, {
    left,
    top: top + height,
    width,
    height: 2,
    fill: "#D0D6DB",
    line: ctx.line(),
  });

  [0, 20, 40, 60, 80, 100].forEach((tick) => {
    const y = top + height - (tick / maxValue) * height;
    ctx.addShape(slide, {
      left,
      top: y,
      width,
      height: tick === 0 ? 0 : 1,
      fill: "#EEF1F4",
      line: ctx.line(),
    });
    ctx.addText(slide, {
      text: String(tick),
      left: left - 34,
      top: y - 8,
      width: 26,
      height: 16,
      fontSize: 9,
      color: STYLE.soft,
      typeface: STYLE.sans,
      align: "right",
    });
  });

  data.forEach((item, index) => {
    const barHeight = (item.correct / maxValue) * height;
    const x = left + index * (barWidth + gap);
    const y = top + height - barHeight;
    const fill = item.label === "base" ? STYLE.accent : item.label === "ft_50" ? STYLE.accent2 : "#8ABFD6";

    ctx.addShape(slide, {
      left: x,
      top: y,
      width: barWidth,
      height: barHeight,
      fill,
      line: ctx.line(),
    });
    ctx.addText(slide, {
      text: String(item.correct),
      left: x - 4,
      top: y - 28,
      width: barWidth + 8,
      height: 22,
      fontSize: 15,
      bold: true,
      color: STYLE.ink,
      typeface: STYLE.sans,
      align: "center",
    });
    ctx.addText(slide, {
      text: item.label,
      left: x - 10,
      top: top + height + 8,
      width: barWidth + 20,
      height: 34,
      fontSize: 11,
      color: STYLE.soft,
      typeface: STYLE.sans,
      align: "center",
    });
  });
}
