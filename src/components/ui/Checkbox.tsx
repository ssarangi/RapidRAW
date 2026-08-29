import React, { ReactNode } from 'react';
import clsx from 'clsx';
import { Check } from 'lucide-react';
import Text from './Text';
import { TextVariants } from '../../types/typography';

export interface CheckboxProps {
  checked: boolean;
  onChange(checked: boolean): void;
  label?: ReactNode;
  description?: ReactNode;
  disabled?: boolean;
  id?: string;
  className?: string;
  boxClassName?: string;
}

/**
 * A standardized checkbox component conforming to the application theme.
 * Uses theme variables (--color-border-color, --color-surface, --color-accent, --color-button-text)
 * to avoid hardcoded borders or browser native styling inconsistencies.
 */
const Checkbox = React.forwardRef<HTMLInputElement, CheckboxProps>(({
  checked,
  onChange,
  label,
  description,
  disabled = false,
  id,
  className = '',
  boxClassName = '',
}, ref) => {
  const uniqueId = id || (typeof label === 'string' ? `checkbox-${label.replace(/\s+/g, '-').toLowerCase()}` : undefined);

  return (
    <label
      htmlFor={uniqueId}
      className={clsx(
        'group inline-flex items-start gap-2.5 select-none transition-colors',
        disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer',
        className,
      )}
    >
      <div className="relative flex items-center justify-center pt-0.5">
        <input
          ref={ref}
          id={uniqueId}
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(e) => !disabled && onChange(e.target.checked)}
          className="sr-only"
        />
        <div
          className={clsx(
            'flex h-4 w-4 shrink-0 items-center justify-center rounded transition-all',
            'border',
            checked
              ? 'border-accent bg-accent text-button-text'
              : 'border-border-color bg-surface/70 group-hover:border-text-secondary',
            boxClassName,
          )}
        >
          {checked && <Check size={12} strokeWidth={3} className="text-current" />}
        </div>
      </div>
      {(label || description) && (
        <div className="min-w-0 flex-1">
          {label && (
            <Text
              as="span"
              variant={TextVariants.small}
              className={clsx(
                'block leading-tight text-text-primary transition-colors',
                !disabled && 'group-hover:text-text-primary',
              )}
            >
              {label}
            </Text>
          )}
          {description && (
            <Text
              as="span"
              variant={TextVariants.extraSmall}
              className="mt-0.5 block text-text-secondary"
            >
              {description}
            </Text>
          )}
        </div>
      )}
    </label>
  );
});

Checkbox.displayName = 'Checkbox';

export default Checkbox;
