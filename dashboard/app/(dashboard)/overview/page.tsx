"use client";

import { useMemo, useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { useChatStore } from "@/stores/chat-store";
import { useTaskStore } from "@/stores/task-store";
import { useConnectionStore } from "@/stores/connection-store";
import { useWebSocket } from "@/components/providers/websocket-provider";
import {
  Users,
  Zap,
  CheckCircle2,
  MessageSquare,
  Activity,
  TrendingUp,
  Wifi,
  WifiOff,
  ArrowUpRight,
  ArrowDownRight,
  Bitcoin,
  RefreshCw,
  Wallet,
  DollarSign,
  Play,
  Square,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatDistanceToNow } from "date-fns";
import { cn } from "@/lib/utils";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface TickerData {
  pair: string;
  price: string;
  change24h: number;
  volume: string;
  high: string;
  low: string;
}

// Live Status Widget - Shows real Kraken balance
function LiveStatusWidget() {
  const [spotBalance, setSpotBalance] = useState<Record<string, string> | null>(null);
  const [futuresBalance, setFuturesBalance] = useState<{ marginEquity: number; availableMargin: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const { connect, isConnected, status } = useWebSocket();

  useEffect(() => {
    async function fetchBalances() {
      try {
        // Fetch Spot balance
        const spotRes = await fetch(`${API_BASE_URL}/api/kraken/balance`);
        if (spotRes.ok) {
          const spotData = await spotRes.json();
          if (spotData.success && spotData.data) {
            const parsed = typeof spotData.data === 'string'
              ? JSON.parse(spotData.data.replace(/'/g, '"'))
              : spotData.data;
            if (parsed.result) {
              setSpotBalance(parsed.result);
            }
          }
        }

        // Fetch Futures balance
        const futuresRes = await fetch(`${API_BASE_URL}/api/kraken/futures/wallets`);
        if (futuresRes.ok) {
          const futuresData = await futuresRes.json();
          if (futuresData.success && futuresData.data) {
            const parsed = typeof futuresData.data === 'string'
              ? JSON.parse(futuresData.data.replace(/'/g, '"'))
              : futuresData.data;
            if (parsed.accounts?.flex) {
              setFuturesBalance({
                marginEquity: parsed.accounts.flex.marginEquity || 0,
                availableMargin: parsed.accounts.flex.availableMargin || 0,
              });
            }
          }
        }
      } catch (error) {
        console.error("Failed to fetch balances:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchBalances();
    // Auto-connect to WebSocket
    if (!isConnected) {
      connect("ws://localhost:8000/ws/spot");
    }
  }, []);

  const totalSpotUSD = spotBalance?.ZUSD ? parseFloat(spotBalance.ZUSD) : 0;
  const totalFuturesUSD = futuresBalance?.marginEquity || 0;

  return (
    <div className="trading-card p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-emerald-500/10 p-2.5 border border-emerald-500/20">
            <Wallet className="h-5 w-5 text-emerald-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold">Live Trading Status</h3>
            <div className="flex items-center gap-2 mt-0.5">
              <span className={cn(
                "h-2 w-2 rounded-full",
                isConnected ? "bg-emerald-500 animate-pulse" : "bg-red-500"
              )} />
              <span className="text-[10px] text-muted-foreground terminal-text">
                {isConnected ? "Connected to Kraken" : "Connecting..."}
              </span>
            </div>
          </div>
        </div>
        <Badge
          variant="secondary"
          className="bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 font-semibold"
        >
          LIVE
        </Badge>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-6">
          <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4">
          {/* Spot Balance */}
          <div className="rounded-lg bg-muted/30 p-4 border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <div className="h-2 w-2 rounded-full bg-orange-500" />
              <span className="text-xs font-medium text-muted-foreground">Spot Account</span>
            </div>
            <p className="text-2xl font-bold terminal-text">
              ${totalSpotUSD.toFixed(2)}
            </p>
            {spotBalance && Object.keys(spotBalance).length > 1 && (
              <p className="text-[10px] text-muted-foreground mt-1">
                + {Object.keys(spotBalance).filter(k => k !== 'ZUSD').length} assets
              </p>
            )}
          </div>

          {/* Futures Balance */}
          <div className="rounded-lg bg-muted/30 p-4 border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <div className="h-2 w-2 rounded-full bg-teal-500" />
              <span className="text-xs font-medium text-muted-foreground">Futures Margin</span>
            </div>
            <p className="text-2xl font-bold terminal-text">
              ${totalFuturesUSD.toFixed(2)}
            </p>
            {futuresBalance && (
              <p className="text-[10px] text-muted-foreground mt-1">
                Available: ${futuresBalance.availableMargin.toFixed(2)}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Kraken Market Widget
function KrakenMarketWidget() {
  const [ticker, setTicker] = useState<TickerData | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchTicker = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/kraken/ticker/XBTUSD`);
      const data = await response.json();

      if (data.success && data.data) {
        // Parse the Kraken ticker response
        const parsed = typeof data.data === 'string' ? JSON.parse(data.data.replace(/'/g, '"')) : data.data;
        const tickerInfo = parsed.result?.XXBTZUSD;

        if (tickerInfo) {
          const currentPrice = parseFloat(tickerInfo.c[0]);
          const openPrice = parseFloat(tickerInfo.o);
          const change24h = ((currentPrice - openPrice) / openPrice) * 100;

          setTicker({
            pair: "BTC/USD",
            price: currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
            change24h: change24h,
            volume: parseFloat(tickerInfo.v[1]).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
            high: parseFloat(tickerInfo.h[1]).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
            low: parseFloat(tickerInfo.l[1]).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
          });
          setLastUpdate(new Date());
        }
      }
    } catch (error) {
      console.error("Failed to fetch ticker:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTicker();
    const interval = setInterval(fetchTicker, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="trading-card p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-orange-500/10 p-2 border border-orange-500/20">
            <Bitcoin className="h-5 w-5 text-orange-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold">Kraken Market Data</h3>
            <p className="text-[10px] text-muted-foreground terminal-text">
              {lastUpdate ? `Updated ${formatDistanceToNow(lastUpdate, { addSuffix: true })}` : "Loading..."}
            </p>
          </div>
        </div>
        <button
          onClick={fetchTicker}
          className="p-1.5 rounded-lg hover:bg-muted/50 transition-colors"
          disabled={loading}
        >
          <RefreshCw className={cn("h-4 w-4 text-muted-foreground", loading && "animate-spin")} />
        </button>
      </div>

      {ticker ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">BTC/USD</p>
            <p className="text-xl font-bold terminal-text">${ticker.price}</p>
            <div className={cn(
              "flex items-center gap-0.5 text-xs font-semibold mt-1",
              ticker.change24h >= 0 ? "text-emerald-400" : "text-red-400"
            )}>
              {ticker.change24h >= 0 ? (
                <ArrowUpRight className="h-3 w-3" />
              ) : (
                <ArrowDownRight className="h-3 w-3" />
              )}
              {Math.abs(ticker.change24h).toFixed(2)}%
            </div>
          </div>
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">24h Volume</p>
            <p className="text-lg font-semibold terminal-text">{ticker.volume}</p>
            <p className="text-[10px] text-muted-foreground">BTC</p>
          </div>
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">24h High</p>
            <p className="text-lg font-semibold terminal-text text-emerald-400">${ticker.high}</p>
          </div>
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">24h Low</p>
            <p className="text-lg font-semibold terminal-text text-red-400">${ticker.low}</p>
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-center py-8 text-muted-foreground text-sm">
          {loading ? "Loading market data..." : "Failed to load market data"}
        </div>
      )}
    </div>
  );
}

// Animated counter component
function AnimatedCounter({ value, duration = 1000 }: { value: number; duration?: number }) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const startTime = Date.now();
    const startValue = displayValue;
    const diff = value - startValue;

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeOut = 1 - Math.pow(1 - progress, 3);
      setDisplayValue(Math.round(startValue + diff * easeOut));

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [value, duration]);

  return <span className="metric-value">{displayValue}</span>;
}

// Stat card component with trading aesthetic
function StatCard({
  name,
  value,
  icon: Icon,
  description,
  color,
  trend,
  delay,
}: {
  name: string;
  value: number;
  icon: React.ElementType;
  description: string;
  color: "blue" | "green" | "emerald" | "amber" | "purple" | "orange" | "teal";
  trend?: { value: number; positive: boolean };
  delay: number;
}) {
  const colorMap = {
    blue: {
      icon: "text-blue-400",
      bg: "bg-blue-500/10",
      border: "border-blue-500/20",
      glow: "group-hover:shadow-blue-500/20",
      gradient: "from-blue-500/20 to-transparent",
    },
    green: {
      icon: "text-green-400",
      bg: "bg-green-500/10",
      border: "border-green-500/20",
      glow: "group-hover:shadow-green-500/20",
      gradient: "from-green-500/20 to-transparent",
    },
    emerald: {
      icon: "text-emerald-400",
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/20",
      glow: "group-hover:shadow-emerald-500/20",
      gradient: "from-emerald-500/20 to-transparent",
    },
    amber: {
      icon: "text-amber-400",
      bg: "bg-amber-500/10",
      border: "border-amber-500/20",
      glow: "group-hover:shadow-amber-500/20",
      gradient: "from-amber-500/20 to-transparent",
    },
    purple: {
      icon: "text-purple-400",
      bg: "bg-purple-500/10",
      border: "border-purple-500/20",
      glow: "group-hover:shadow-purple-500/20",
      gradient: "from-purple-500/20 to-transparent",
    },
    orange: {
      icon: "text-orange-400",
      bg: "bg-orange-500/10",
      border: "border-orange-500/20",
      glow: "group-hover:shadow-orange-500/20",
      gradient: "from-orange-500/20 to-transparent",
    },
    teal: {
      icon: "text-teal-400",
      bg: "bg-teal-500/10",
      border: "border-teal-500/20",
      glow: "group-hover:shadow-teal-500/20",
      gradient: "from-teal-500/20 to-transparent",
    },
  };

  const colors = colorMap[color];

  return (
    <div
      className={cn(
        "group stat-card p-5",
        "hover:shadow-xl transition-all duration-300",
        colors.glow
      )}
    >
      {/* Top gradient line */}
      <div className={cn(
        "absolute top-0 left-0 right-0 h-px bg-gradient-to-r",
        colors.gradient
      )} />

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          {name}
        </span>
        <div className={cn(
          "rounded-lg p-2 transition-all duration-300",
          colors.bg,
          "border",
          colors.border,
          "group-hover:scale-110"
        )}>
          <Icon className={cn("h-4 w-4", colors.icon)} />
        </div>
      </div>

      {/* Value */}
      <div className="flex items-end gap-3">
        <div className="text-4xl font-bold terminal-text tracking-tight">
          <AnimatedCounter value={value} />
        </div>
        {trend && (
          <div className={cn(
            "flex items-center gap-0.5 text-xs font-semibold mb-1.5",
            trend.positive ? "text-emerald-400" : "text-red-400"
          )}>
            {trend.positive ? (
              <ArrowUpRight className="h-3 w-3" />
            ) : (
              <ArrowDownRight className="h-3 w-3" />
            )}
            {trend.value}%
          </div>
        )}
      </div>

      {/* Description */}
      <p className="text-xs text-muted-foreground mt-2 terminal-text">
        {description}
      </p>
    </div>
  );
}

// Crew progress bar component with Start/Stop
function CrewProgress({
  name,
  crewId,
  color,
  agents,
  active,
  status,
  onStart,
  onStop,
  isLoading,
  delay,
}: {
  name: string;
  crewId: string;
  color: "staff" | "spot" | "futures";
  agents: number;
  active: number;
  status: "running" | "stopped";
  onStart: () => void;
  onStop: () => void;
  isLoading: boolean;
  delay: number;
}) {
  const percentage = status === "running" ? Math.min((active / agents) * 100 + 30, 100) : 10;

  const colorConfig = {
    staff: {
      dot: "bg-purple-500",
      progress: "progress-staff",
      badge: "bg-purple-500/15 text-purple-400 border-purple-500/30",
      button: "bg-purple-500 hover:bg-purple-600",
    },
    spot: {
      dot: "bg-orange-500",
      progress: "progress-spot",
      badge: "bg-orange-500/15 text-orange-400 border-orange-500/30",
      button: "bg-orange-500 hover:bg-orange-600",
    },
    futures: {
      dot: "bg-teal-500",
      progress: "progress-futures",
      badge: "bg-teal-500/15 text-teal-400 border-teal-500/30",
      button: "bg-teal-500 hover:bg-teal-600",
    },
  };

  const config = colorConfig[color];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className={cn("h-3 w-3 rounded-full", config.dot)} />
            {status === "running" && (
              <div className={cn(
                "absolute inset-0 h-3 w-3 rounded-full animate-ping",
                config.dot,
                "opacity-75"
              )} />
            )}
          </div>
          <span className="text-sm font-semibold text-foreground">{name}</span>
          {status === "running" ? (
            <Badge
              variant="secondary"
              className="text-[10px] font-semibold border bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
            >
              Running
            </Badge>
          ) : (
            <Badge
              variant="secondary"
              className="text-[10px] font-semibold border bg-muted text-muted-foreground"
            >
              Stopped
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-muted-foreground terminal-text">
            {agents} agents
          </span>
          {status === "running" ? (
            <Button
              size="sm"
              variant="destructive"
              onClick={onStop}
              disabled={isLoading}
              className="h-7 px-2 text-xs"
            >
              {isLoading ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <>
                  <Square className="h-3 w-3 mr-1" />
                  Stop
                </>
              )}
            </Button>
          ) : (
            <Button
              size="sm"
              onClick={onStart}
              disabled={isLoading}
              className={cn("h-7 px-2 text-xs text-white", config.button)}
            >
              {isLoading ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <>
                  <Play className="h-3 w-3 mr-1" />
                  Start
                </>
              )}
            </Button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted/50">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-1000 ease-out",
            config.progress
          )}
          style={{ width: `${percentage}%` }}
        />
        {/* Shimmer effect */}
        {status === "running" && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-shimmer" />
        )}
      </div>
    </div>
  );
}

export default function OverviewPage() {
  const { messages } = useChatStore();
  const { tasks, runningTaskIds, completedTaskIds } = useTaskStore();
  const status = useConnectionStore((s) => s.status);
  const [totalAgents, setTotalAgents] = useState(74);
  const [crewCounts, setCrewCounts] = useState({ staff: 10, spot: 32, futures: 32 });
  const [crewStatuses, setCrewStatuses] = useState<Record<string, "running" | "stopped">>({
    staff: "stopped",
    spot: "stopped",
    futures: "stopped",
  });
  const [loadingCrews, setLoadingCrews] = useState<Record<string, boolean>>({});

  // Fetch real agent counts and statuses from API
  useEffect(() => {
    async function fetchCrews() {
      try {
        const res = await fetch(`${API_BASE_URL}/api/crews`);
        if (res.ok) {
          const crews = await res.json();
          const total = crews.reduce((sum: number, c: { agent_count: number }) => sum + c.agent_count, 0);
          setTotalAgents(total);
          const counts: Record<string, number> = {};
          const statuses: Record<string, "running" | "stopped"> = {};
          crews.forEach((c: { id: string; agent_count: number; status: string }) => {
            counts[c.id] = c.agent_count;
            statuses[c.id] = c.status === "running" ? "running" : "stopped";
          });
          setCrewCounts({
            staff: counts.staff || 10,
            spot: counts.spot || 32,
            futures: counts.futures || 32,
          });
          setCrewStatuses(statuses);
        }
      } catch (error) {
        console.error("Failed to fetch crews:", error);
      }
    }
    fetchCrews();
    // Poll every 5 seconds for status updates
    const interval = setInterval(fetchCrews, 5000);
    return () => clearInterval(interval);
  }, []);

  // Start a crew
  const startCrew = async (crewId: string) => {
    setLoadingCrews((prev) => ({ ...prev, [crewId]: true }));
    try {
      const res = await fetch(`${API_BASE_URL}/api/crews/${crewId}/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ inputs: {} }),
      });
      if (res.ok) {
        setCrewStatuses((prev) => ({ ...prev, [crewId]: "running" }));
      } else {
        const error = await res.json();
        console.error("Failed to start crew:", error);
      }
    } catch (error) {
      console.error("Failed to start crew:", error);
    } finally {
      setLoadingCrews((prev) => ({ ...prev, [crewId]: false }));
    }
  };

  // Stop a crew
  const stopCrew = async (crewId: string) => {
    setLoadingCrews((prev) => ({ ...prev, [crewId]: true }));
    try {
      const res = await fetch(`${API_BASE_URL}/api/crews/${crewId}/stop`, {
        method: "POST",
      });
      if (res.ok) {
        setCrewStatuses((prev) => ({ ...prev, [crewId]: "stopped" }));
      } else {
        const error = await res.json();
        console.error("Failed to stop crew:", error);
      }
    } catch (error) {
      console.error("Failed to stop crew:", error);
    } finally {
      setLoadingCrews((prev) => ({ ...prev, [crewId]: false }));
    }
  };

  // Calculate stats from real data
  const activeAgents = new Set(runningTaskIds.map(id => tasks.get(id)?.agentId).filter(Boolean)).size;
  const completedToday = completedTaskIds.length;
  const totalMessages = messages.length;

  // Get recent activity from messages
  const recentActivity = useMemo(() => {
    return messages
      .slice(-5)
      .reverse()
      .map((msg) => ({
        agent: msg.agentRole,
        action: msg.content.slice(0, 50) + (msg.content.length > 50 ? "..." : ""),
        time: formatDistanceToNow(new Date(msg.timestamp), { addSuffix: true }),
        status: msg.isStreaming ? "running" : "completed",
      }));
  }, [messages]);

  // Calculate running tasks per desk
  const deskStats = useMemo(() => {
    let spotActive = 0;
    let futuresActive = 0;
    let staffActive = 0;

    runningTaskIds.forEach((id) => {
      const task = tasks.get(id);
      if (!task) return;
      const role = task.agentRole.toLowerCase();
      if (role.includes("ceo") || role.includes("group") || role.includes("gcio") || role.includes("gcro")) {
        staffActive++;
      } else if (role.includes("futures") || role.includes("carry") || role.includes("microstructure")) {
        futuresActive++;
      } else {
        spotActive++;
      }
    });

    return { spotActive, futuresActive, staffActive };
  }, [runningTaskIds, tasks]);

  const stats = [
    {
      name: "Total Agents",
      value: totalAgents,
      icon: Users,
      description: `${crewCounts.staff} STAFF + ${crewCounts.spot} Spot + ${crewCounts.futures} Futures`,
      color: "blue" as const,
    },
    {
      name: "Active Now",
      value: activeAgents || 0,
      icon: Zap,
      description: "Currently executing tasks",
      color: "green" as const,
      trend: activeAgents > 0 ? { value: 12, positive: true } : undefined,
    },
    {
      name: "Tasks Completed",
      value: completedToday || 0,
      icon: CheckCircle2,
      description: "This session",
      color: "emerald" as const,
    },
    {
      name: "Messages",
      value: totalMessages || 0,
      icon: MessageSquare,
      description: "Agent communications",
      color: "amber" as const,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Live Trading Status */}
      <LiveStatusWidget />

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat, index) => (
          <StatCard
            key={stat.name}
            {...stat}
            delay={index * 100}
          />
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Crew Status */}
        <div className="trading-card p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="rounded-lg bg-primary/10 p-2 border border-primary/20">
              <Activity className="h-5 w-5 text-primary" />
            </div>
            <h2 className="text-lg font-bold">Crew Status</h2>
          </div>

          <div className="space-y-5">
            <CrewProgress
              name="STAFF"
              crewId="staff"
              color="staff"
              agents={crewCounts.staff}
              active={deskStats.staffActive}
              status={crewStatuses.staff || "stopped"}
              onStart={() => startCrew("staff")}
              onStop={() => stopCrew("staff")}
              isLoading={loadingCrews.staff || false}
              delay={500}
            />
            <CrewProgress
              name="Spot Desk"
              crewId="spot"
              color="spot"
              agents={crewCounts.spot}
              active={deskStats.spotActive}
              status={crewStatuses.spot || "stopped"}
              onStart={() => startCrew("spot")}
              onStop={() => stopCrew("spot")}
              isLoading={loadingCrews.spot || false}
              delay={600}
            />
            <CrewProgress
              name="Futures Desk"
              crewId="futures"
              color="futures"
              agents={crewCounts.futures}
              active={deskStats.futuresActive}
              status={crewStatuses.futures || "stopped"}
              onStart={() => startCrew("futures")}
              onStop={() => stopCrew("futures")}
              isLoading={loadingCrews.futures || false}
              delay={700}
            />
          </div>
        </div>

        {/* Recent Activity */}
        <div className="trading-card p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="rounded-lg bg-emerald-500/10 p-2 border border-emerald-500/20">
              <TrendingUp className="h-5 w-5 text-emerald-400" />
            </div>
            <h2 className="text-lg font-bold">Recent Activity</h2>
          </div>

          {recentActivity.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="rounded-full bg-muted/50 p-4 mb-4">
                <Activity className="h-8 w-8 text-muted-foreground/50" />
              </div>
              <p className="text-sm font-medium text-muted-foreground">No recent activity</p>
              <p className="text-xs text-muted-foreground/70 mt-1">
                Start a crew task to see activity
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {recentActivity.map((activity, index) => (
                <div
                  key={index}
                  className="flex items-start gap-3 text-sm"
                >
                  <div className="relative mt-1">
                    <div
                      className={cn(
                        "h-2.5 w-2.5 rounded-full flex-shrink-0",
                        activity.status === "running"
                          ? "bg-blue-500"
                          : "bg-emerald-500"
                      )}
                    />
                    {activity.status === "running" && (
                      <div className="absolute inset-0 h-2.5 w-2.5 rounded-full bg-blue-500 animate-ping opacity-75" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0 space-y-0.5">
                    <p className="font-semibold text-foreground">{activity.agent}</p>
                    <p className="text-muted-foreground truncate text-xs">{activity.action}</p>
                  </div>
                  <span className="text-[10px] text-muted-foreground/70 flex-shrink-0 terminal-text">
                    {activity.time}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Connection Status */}
      <div className="trading-card p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={cn(
              "rounded-lg p-2 border transition-colors",
              status === "connected"
                ? "bg-emerald-500/10 border-emerald-500/20"
                : "bg-muted/50 border-border"
            )}>
              {status === "connected" ? (
                <Wifi className="h-4 w-4 text-emerald-400" />
              ) : (
                <WifiOff className="h-4 w-4 text-muted-foreground" />
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold">Connection Status</h3>
              <p className="text-xs text-muted-foreground mt-0.5">
                {status === "connected"
                  ? "WebSocket connection is active. Receiving real-time updates."
                  : status === "connecting"
                  ? "Establishing connection to server..."
                  : "Connecting to trading server..."}
              </p>
            </div>
          </div>
          <Badge
            variant="secondary"
            className={cn(
              "font-semibold border",
              status === "connected"
                ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                : status === "connecting"
                ? "bg-amber-500/15 text-amber-400 border-amber-500/30"
                : "bg-muted text-muted-foreground border-border"
            )}
          >
            {status === "connected" && (
              <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_6px_rgba(52,211,153,0.8)]" />
            )}
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </Badge>
        </div>
      </div>

      {/* Kraken Market Data */}
      <KrakenMarketWidget />
    </div>
  );
}
