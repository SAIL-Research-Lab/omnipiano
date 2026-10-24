import { useEffect, useState } from "react";

export function Field({ label, hint, children, wide = false }) {
  return (
    <label className={`field ${wide ? "field--wide" : ""}`}>
      <span className="field__label">{label}</span>
      {children}
      {hint && <span className="field__hint">{hint}</span>}
    </label>
  );
}

export function TextField({ label, value, onChange, hint, wide, multiline }) {
  const Element = multiline ? "textarea" : "input";
  return (
    <Field label={label} hint={hint} wide={wide}>
      <Element
        className="control"
        value={value ?? ""}
        rows={multiline ? 3 : undefined}
        onChange={(event) => onChange(event.target.value)}
      />
    </Field>
  );
}

export function ListField({ label, values, parse, onChange, hint, wide }) {
  const serialized = values.join(", ");
  const [draft, setDraft] = useState(serialized);
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    if (!editing) setDraft(serialized);
  }, [editing, serialized]);

  return (
    <Field label={label} hint={hint} wide={wide}>
      <input
        className="control"
        value={draft}
        onFocus={() => setEditing(true)}
        onChange={(event) => {
          const raw = event.target.value;
          setDraft(raw);
          onChange(parse(raw));
        }}
        onBlur={() => {
          setEditing(false);
          setDraft(parse(draft).join(", "));
        }}
      />
    </Field>
  );
}

export function NumberInput({ value, onChange, min, max, step = 1, ...props }) {
  const serialized = value == null ? "" : String(value);
  const [draft, setDraft] = useState(serialized);
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    if (!editing) setDraft(serialized);
  }, [editing, serialized]);

  return (
    <input
      {...props}
      className={`control ${props.className ?? ""}`.trim()}
      type="number"
      value={draft}
      min={min}
      max={max}
      step={step}
      onFocus={(event) => {
        setEditing(true);
        props.onFocus?.(event);
      }}
      onChange={(event) => {
        const raw = event.target.value;
        setDraft(raw);
        if (raw === "") return;
        const parsed = Number(raw);
        if (Number.isFinite(parsed)) onChange(parsed);
      }}
      onBlur={(event) => {
        setEditing(false);
        setDraft(serialized);
        props.onBlur?.(event);
      }}
    />
  );
}

export function NumberField({ label, value, onChange, hint, min, max, step = 1 }) {
  return (
    <Field label={label} hint={hint}>
      <NumberInput
        value={value ?? ""}
        min={min}
        max={max}
        step={step}
        onChange={onChange}
      />
    </Field>
  );
}

export function SelectField({ label, value, onChange, options, hint, disabled = false }) {
  return (
    <Field label={label} hint={hint}>
      <select
        className="control"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((option) => {
          const normalized = typeof option === "string"
            ? { value: option, label: option.replaceAll("_", " ") }
            : option;
          return (
            <option value={normalized.value} key={normalized.value}>
              {normalized.label}
            </option>
          );
        })}
      </select>
    </Field>
  );
}

export function ToggleField({ label, checked, onChange, hint }) {
  return (
    <Field label={label} hint={hint}>
      <button
        type="button"
        className={`toggle ${checked ? "toggle--on" : ""}`}
        aria-pressed={checked}
        onClick={() => onChange(!checked)}
      >
        <span className="toggle__knob" />
        <span>{checked ? "Enabled" : "Disabled"}</span>
      </button>
    </Field>
  );
}

export function ChoiceRow({ label, value, options, onChange, wide = true }) {
  return (
    <div className={`choice-block ${wide ? "field--wide" : ""}`.trim()}>
      <span className="field__label">{label}</span>
      <div className="choice-row">
        {options.map((option) => {
          const item = typeof option === "string"
            ? { value: option, label: option.replaceAll("_", " ") }
            : option;
          return (
            <button
              type="button"
              className={value === item.value ? "active" : ""}
              aria-pressed={value === item.value}
              key={item.value}
              onClick={() => onChange(item.value)}
            >
              {item.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function Section({
  eyebrow,
  title,
  description,
  children,
  advanced,
  onToggleAdvanced,
  hideTitle = false,
  advancedAtBottom = false,
}) {
  const modeButton = onToggleAdvanced && (
    <button
      type="button"
      className={`panel-mode ${advanced ? "panel-mode--active" : ""}`}
      aria-pressed={advanced}
      aria-label={`${advanced ? "Hide" : "Show"} advanced settings for ${title}`}
      onClick={onToggleAdvanced}
    >
      {advanced ? "Basic" : "Advanced"}
    </button>
  );
  const showHeading = !hideTitle && (eyebrow || title || description || (modeButton && !advancedAtBottom));
  return (
    <section className="section-card">
      {showHeading && (
        <div className="section-card__heading">
          <div className="section-card__title-row">
            <div>
              {eyebrow && <span className="eyebrow">{eyebrow}</span>}
              {title && <h2>{title}</h2>}
            </div>
            {!advancedAtBottom && modeButton}
          </div>
          {description && <p>{description}</p>}
        </div>
      )}
      <div className="field-grid">{children}</div>
      {advancedAtBottom && modeButton && <div className="section-card__footer">{modeButton}</div>}
    </section>
  );
}
