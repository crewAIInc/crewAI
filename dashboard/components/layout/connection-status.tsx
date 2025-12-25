"use client";

import { useConnectionStore, type ConnectionStatus } from "@/stores/connection-store";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Wifi, WifiOff, Loader2, AlertCircle } from "lucide-react";

const statusConfig: Record<
  ConnectionStatus,
  { color: string; label: string; Icon: typeof Wifi; animate?: boolean }
> = {
  connected: {
    color: "bg-green-500",
    label: "Connected",
    Icon: Wifi,
  },
  connecting: {
    color: "bg-yellow-500",
    label: "Connecting...",
    Icon: Loader2,
    animate: true,
  },
  disconnected: {
    color: "bg-gray-400",
    label: "Disconnected",
    Icon: WifiOff,
  },
  error: {
    color: "bg-red-500",
    label: "Connection Error",
    Icon: AlertCircle,
  },
};

interface ConnectionStatusProps {
  showLabel?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function ConnectionStatus({
  showLabel = false,
  size = "md",
  className,
}: ConnectionStatusProps) {
  const { status, error, reconnectAttempts } = useConnectionStore();
  const config = statusConfig[status];

  const sizeClasses = {
    sm: "h-2 w-2",
    md: "h-2.5 w-2.5",
    lg: "h-3 w-3",
  };

  const iconSizes = {
    sm: "h-3 w-3",
    md: "h-4 w-4",
    lg: "h-5 w-5",
  };

  const tooltipContent = (
    <div className="text-xs">
      <div className="font-medium">{config.label}</div>
      {error && <div className="text-red-400 mt-1">{error}</div>}
      {status === "connecting" && reconnectAttempts > 0 && (
        <div className="text-muted-foreground mt-1">
          Attempt {reconnectAttempts}/5
        </div>
      )}
    </div>
  );

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={cn(
              "flex items-center gap-2 cursor-default",
              className
            )}
          >
            <div className="relative">
              <div
                className={cn(
                  "rounded-full transition-colors",
                  sizeClasses[size],
                  config.color,
                  status === "connected" && "animate-pulse"
                )}
              />
              {status === "connecting" && (
                <div
                  className={cn(
                    "absolute inset-0 rounded-full animate-ping",
                    config.color,
                    "opacity-50"
                  )}
                />
              )}
            </div>
            {showLabel && (
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <config.Icon
                  className={cn(
                    iconSizes[size],
                    config.animate && "animate-spin"
                  )}
                />
                <span>{config.label}</span>
              </div>
            )}
          </div>
        </TooltipTrigger>
        <TooltipContent side="right">{tooltipContent}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// Compact version for header/sidebar
export function ConnectionIndicator({ className }: { className?: string }) {
  const status = useConnectionStore((s) => s.status);
  const config = statusConfig[status];

  return (
    <div className={cn("flex items-center gap-1.5", className)}>
      <div
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          config.color,
          status === "connected" && "animate-pulse"
        )}
      />
      <span className="text-[10px] text-muted-foreground uppercase tracking-wide">
        {status === "connected"
          ? "Live"
          : status === "connecting"
          ? "..."
          : "Offline"}
      </span>
    </div>
  );
}
