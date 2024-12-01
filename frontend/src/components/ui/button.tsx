import * as React from "react"
import { cn } from "../../lib/utils"

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'secondary' | 'outline' | 'ghost';
  size?: 'default' | 'sm' | 'lg';
  children: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', children, ...props }, ref) => {
    return (
      <button
        className={cn(
          // Base styles
          "inline-flex items-center justify-center rounded-md font-medium transition-colors",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2",
          "disabled:opacity-50 disabled:pointer-events-none",
          
          // Variant styles
          variant === 'default' && "bg-blue-600 text-white hover:bg-blue-700",
          variant === 'secondary' && "bg-gray-100 text-gray-900 hover:bg-gray-200",
          variant === 'outline' && "border border-gray-300 bg-transparent hover:bg-gray-100",
          variant === 'ghost' && "bg-transparent hover:bg-gray-100",
          
          // Size styles
          size === 'default' && "h-10 px-4 py-2",
          size === 'sm' && "h-8 px-3 text-sm",
          size === 'lg' && "h-12 px-8",
          
          className
        )}
        ref={ref}
        {...props}
      >
        {children}
      </button>
    )
  }
)

Button.displayName = "Button" 