"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";
import { ChevronRight, Wrench, CheckCircle2, XCircle, Clock } from "lucide-react";
import type { ToolCallInfo } from "@/lib/types/stream";

interface ToolCallCardProps {
  toolCall: ToolCallInfo;
  className?: string;
}

export function ToolCallCard({ toolCall, className }: ToolCallCardProps) {
  const [isOpen, setIsOpen] = useState(false);

  const formatArguments = (args: string) => {
    try {
      const parsed = JSON.parse(args);
      return JSON.stringify(parsed, null, 2);
    } catch {
      return args;
    }
  };

  const formatResult = (result: string | undefined) => {
    if (!result) return null;
    try {
      const parsed = JSON.parse(result);
      return JSON.stringify(parsed, null, 2);
    } catch {
      return result;
    }
  };

  const hasResult = toolCall.result && toolCall.result !== "";

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger asChild>
        <button
          className={cn(
            "w-full rounded-lg border bg-muted/30 p-3 text-left transition-all",
            "hover:bg-muted/50 hover:border-primary/30",
            isOpen && "border-primary/30 bg-muted/50",
            className
          )}
        >
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "flex h-8 w-8 items-center justify-center rounded-lg",
                toolCall.success !== false
                  ? "bg-blue-500/10 text-blue-500"
                  : "bg-red-500/10 text-red-500"
              )}
            >
              <Wrench className="h-4 w-4" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm font-medium truncate">
                  {toolCall.toolName}
                </span>
                {hasResult ? (
                  toolCall.success !== false ? (
                    <Badge
                      variant="secondary"
                      className="bg-green-500/10 text-green-500 text-[10px]"
                    >
                      <CheckCircle2 className="h-3 w-3 mr-1" />
                      Success
                    </Badge>
                  ) : (
                    <Badge
                      variant="secondary"
                      className="bg-red-500/10 text-red-500 text-[10px]"
                    >
                      <XCircle className="h-3 w-3 mr-1" />
                      Failed
                    </Badge>
                  )
                ) : (
                  <Badge
                    variant="secondary"
                    className="bg-yellow-500/10 text-yellow-500 text-[10px]"
                  >
                    <Clock className="h-3 w-3 mr-1" />
                    Running
                  </Badge>
                )}
              </div>
              {toolCall.toolId && (
                <span className="text-xs text-muted-foreground font-mono">
                  {toolCall.toolId}
                </span>
              )}
            </div>
            <ChevronRight
              className={cn(
                "h-4 w-4 text-muted-foreground transition-transform",
                isOpen && "rotate-90"
              )}
            />
          </div>
        </button>
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="mt-2 space-y-3 rounded-lg border bg-muted/20 p-3">
          {/* Arguments */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Arguments
              </span>
            </div>
            <div className="code-block">
              <pre className="text-xs overflow-x-auto">
                <code>{formatArguments(toolCall.arguments)}</code>
              </pre>
            </div>
          </div>

          {/* Result */}
          {hasResult && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Result
                </span>
              </div>
              <div
                className={cn(
                  "code-block",
                  toolCall.success === false && "border-red-500/30"
                )}
              >
                <pre className="text-xs overflow-x-auto max-h-48">
                  <code>{formatResult(toolCall.result)}</code>
                </pre>
              </div>
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
