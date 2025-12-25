"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  MessageSquare,
  Kanban,
  Users,
  Users2,
  BarChart3,
  Settings,
  Zap,
  Activity,
  Wrench,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ConnectionStatus } from "@/components/layout/connection-status";

const API_BASE = "http://localhost:8000";

const navigation = [
  {
    name: "Overview",
    href: "/overview",
    icon: LayoutDashboard,
  },
  {
    name: "Chat",
    href: "/chat",
    icon: MessageSquare,
    badge: "Live",
  },
  {
    name: "Tasks",
    href: "/tasks",
    icon: Kanban,
  },
  {
    name: "Agents",
    href: "/agents",
    icon: Users,
  },
  {
    name: "Crews",
    href: "/crews",
    icon: Users2,
  },
  {
    name: "Tools",
    href: "/tools",
    icon: Wrench,
  },
  {
    name: "Metrics",
    href: "/metrics",
    icon: BarChart3,
  },
];

interface CrewData {
  id: string;
  name: string;
  agent_count: number;
}

const crewStyles: Record<string, { color: string; glowColor: string }> = {
  staff: { color: "bg-purple-500", glowColor: "shadow-purple-500/50" },
  spot: { color: "bg-orange-500", glowColor: "shadow-orange-500/50" },
  futures: { color: "bg-teal-500", glowColor: "shadow-teal-500/50" },
};

export function Sidebar() {
  const pathname = usePathname();
  const [crews, setCrews] = useState<CrewData[]>([]);
  const [totalAgents, setTotalAgents] = useState(0);

  useEffect(() => {
    async function fetchCrews() {
      try {
        const res = await fetch(`${API_BASE}/api/crews`);
        if (res.ok) {
          const data: CrewData[] = await res.json();
          setCrews(data);
          setTotalAgents(data.reduce((sum, c) => sum + c.agent_count, 0));
        }
      } catch (error) {
        console.error("Failed to fetch crews:", error);
        // Fallback data
        setCrews([
          { id: "staff", name: "STAFF", agent_count: 10 },
          { id: "spot", name: "Spot", agent_count: 32 },
          { id: "futures", name: "Futures", agent_count: 32 },
        ]);
        setTotalAgents(74);
      }
    }
    fetchCrews();
  }, []);

  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-64 border-r border-border/50">
      {/* Glassmorphism background */}
      <div className="absolute inset-0 bg-sidebar/90 backdrop-blur-xl" />

      {/* Right edge glow */}
      <div className="absolute top-0 right-0 bottom-0 w-px bg-gradient-to-b from-transparent via-primary/30 to-transparent" />

      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-primary/5 via-transparent to-transparent pointer-events-none" />

      <div className="relative flex h-full flex-col">
        {/* Logo Section */}
        <div className="flex h-16 items-center gap-3 border-b border-border/50 px-6">
          <div className="relative">
            {/* Animated glow ring */}
            <div className="absolute -inset-1 rounded-xl logo-gradient opacity-75 blur-sm" />
            <div className="relative flex h-9 w-9 items-center justify-center rounded-xl logo-gradient shadow-lg">
              <Zap className="h-5 w-5 text-white drop-shadow-lg" />
            </div>
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-bold tracking-tight text-foreground">
              QRI Trading
            </span>
            <div className="flex items-center gap-1.5">
              <Activity className="h-3 w-3 text-emerald-500 animate-pulse" />
              <span className="text-[10px] font-medium text-muted-foreground terminal-text">
                {totalAgents} Agents
              </span>
            </div>
          </div>
        </div>

        <ScrollArea className="flex-1 px-3 py-4">
          {/* Main Navigation */}
          <nav className="space-y-1">
            {navigation.map((item, index) => {
              const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
                    isActive
                      ? "nav-active text-foreground"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                  )}
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <item.icon
                    className={cn(
                      "h-4 w-4 shrink-0 transition-all duration-200",
                      isActive
                        ? "text-primary drop-shadow-[0_0_8px_rgba(59,130,246,0.5)]"
                        : "text-muted-foreground group-hover:text-primary"
                    )}
                  />
                  <span className="flex-1">{item.name}</span>
                  {item.badge && (
                    <Badge
                      variant="secondary"
                      className="h-5 bg-emerald-500/15 text-emerald-400 text-[10px] font-semibold border border-emerald-500/30"
                    >
                      <span className="mr-1.5 h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]" />
                      {item.badge}
                    </Badge>
                  )}
                </Link>
              );
            })}
          </nav>

          <Separator className="my-4 bg-border/50" />

          {/* Crews Section */}
          <div className="space-y-3">
            <h3 className="px-3 text-[10px] font-bold uppercase tracking-widest text-muted-foreground/70">
              Crews
            </h3>
            <div className="space-y-1">
              {crews.map((crew) => {
                const style = crewStyles[crew.id] || { color: "bg-gray-500", glowColor: "shadow-gray-500/50" };
                return (
                  <button
                    key={crew.id}
                    className="group flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground transition-all duration-200 hover:bg-muted/50 hover:text-foreground"
                  >
                    <div className="relative">
                      <div className={cn(
                        "h-2.5 w-2.5 rounded-full transition-all duration-300",
                        style.color,
                        "group-hover:shadow-lg",
                        style.glowColor
                      )} />
                      <div className={cn(
                        "absolute inset-0 h-2.5 w-2.5 rounded-full animate-ping opacity-0 group-hover:opacity-75",
                        style.color
                      )} />
                    </div>
                    <span className="flex-1 text-left">{crew.name}</span>
                    <span className="terminal-text text-xs font-semibold text-muted-foreground/70 tabular-nums">
                      {crew.agent_count}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          <Separator className="my-4 bg-border/50" />

          {/* Connection Status */}
          <div className="rounded-xl bg-muted/30 p-3 border border-border/50">
            <ConnectionStatus showLabel size="md" />
            <p className="mt-2 text-[10px] text-muted-foreground/70 terminal-text">
              WebSocket status
            </p>
          </div>
        </ScrollArea>

        {/* Bottom Section */}
        <div className="border-t border-border/50 p-3">
          <Link
            href="/settings"
            className={cn(
              "group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
              pathname === "/settings"
                ? "nav-active text-foreground"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
            )}
          >
            <Settings className={cn(
              "h-4 w-4 transition-all duration-300",
              pathname === "/settings"
                ? "text-primary"
                : "text-muted-foreground group-hover:text-primary group-hover:rotate-90"
            )} />
            <span>Settings</span>
          </Link>
        </div>
      </div>
    </aside>
  );
}
