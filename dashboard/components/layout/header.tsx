"use client";

import { usePathname } from "next/navigation";
import { Search, Bell, Command } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ThemeToggle } from "./theme-toggle";
import { useTaskStore } from "@/stores/task-store";

const pageNames: Record<string, string> = {
  "/overview": "Overview",
  "/chat": "Chat Timeline",
  "/tasks": "Task Board",
  "/agents": "Agents",
  "/metrics": "Metrics",
  "/settings": "Settings",
};

export function Header() {
  const pathname = usePathname();
  const pageName = pageNames[pathname] || "Dashboard";
  const { runningTaskIds, completedTaskIds } = useTaskStore();

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-background/80 px-6 backdrop-blur-sm">
      {/* Left: Page Title + Breadcrumb */}
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-semibold text-foreground">{pageName}</h1>
        {pathname === "/chat" && (
          <Badge variant="secondary" className="gap-1 bg-green-500/10 text-green-500">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-green-500" />
            Live
          </Badge>
        )}
      </div>

      {/* Center: Search */}
      <div className="flex flex-1 max-w-md mx-8">
        <div className="relative w-full">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search agents, tasks..."
            className="w-full bg-muted/50 pl-9 pr-12 border-0 focus-visible:ring-1"
          />
          <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
            <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
              <Command className="h-3 w-3" />K
            </kbd>
          </div>
        </div>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-2">
        {/* Live Stats */}
        <div className="hidden md:flex items-center gap-4 mr-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <span className={`h-2 w-2 rounded-full ${runningTaskIds.length > 0 ? 'bg-blue-500 animate-pulse' : 'bg-muted'}`} />
            <span>{runningTaskIds.length} running</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-green-500" />
            <span>{completedTaskIds.length} completed</span>
          </div>
        </div>

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="relative">
          <Bell className="h-4 w-4" />
          <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-red-500" />
        </Button>

        {/* Theme Toggle */}
        <ThemeToggle />
      </div>
    </header>
  );
}
