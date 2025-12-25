"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  BarChart3,
  TrendingUp,
  Clock,
  Zap,
  Target,
  Activity,
} from "lucide-react";

const metrics = [
  {
    name: "Total Tasks Today",
    value: "247",
    change: "+12%",
    trend: "up",
    icon: Activity,
  },
  {
    name: "Success Rate",
    value: "98.3%",
    change: "+0.5%",
    trend: "up",
    icon: Target,
  },
  {
    name: "Avg. Duration",
    value: "2.1s",
    change: "-0.3s",
    trend: "up",
    icon: Clock,
  },
  {
    name: "Tokens Used",
    value: "1.2M",
    change: "+8%",
    trend: "neutral",
    icon: Zap,
  },
];

const crewMetrics = [
  {
    name: "STAFF",
    tasks: 45,
    success: 100,
    avgDuration: 1.8,
    color: "bg-purple-500",
  },
  {
    name: "Spot",
    tasks: 124,
    success: 97.5,
    avgDuration: 2.3,
    color: "bg-orange-500",
  },
  {
    name: "Futures",
    tasks: 78,
    success: 98.7,
    avgDuration: 2.0,
    color: "bg-teal-500",
  },
];

const hourlyData = [
  { hour: "00:00", tasks: 12 },
  { hour: "04:00", tasks: 8 },
  { hour: "08:00", tasks: 45 },
  { hour: "12:00", tasks: 62 },
  { hour: "16:00", tasks: 78 },
  { hour: "20:00", tasks: 42 },
];

export default function MetricsPage() {
  const maxTasks = Math.max(...hourlyData.map((d) => d.tasks));

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Metrics & Analytics
          </h2>
          <p className="text-sm text-muted-foreground">
            Real-time performance monitoring
          </p>
        </div>
        <Badge variant="outline">Last 24 hours</Badge>
      </div>

      {/* Top Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {metrics.map((metric, index) => (
          <Card
            key={metric.name}
            className="animate-scale-in"
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {metric.name}
              </CardTitle>
              <metric.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold">{metric.value}</span>
                <span
                  className={`text-xs font-medium ${
                    metric.trend === "up"
                      ? "text-green-500"
                      : metric.trend === "down"
                      ? "text-red-500"
                      : "text-muted-foreground"
                  }`}
                >
                  {metric.change}
                </span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Hourly Tasks Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">
              Tasks by Hour
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-end justify-between h-48 gap-2">
              {hourlyData.map((data, index) => (
                <div
                  key={data.hour}
                  className="flex-1 flex flex-col items-center gap-2"
                >
                  <div
                    className="w-full bg-primary/80 rounded-t transition-all hover:bg-primary animate-scale-in"
                    style={{
                      height: `${(data.tasks / maxTasks) * 100}%`,
                      animationDelay: `${index * 100}ms`,
                    }}
                  />
                  <span className="text-[10px] text-muted-foreground">
                    {data.hour}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Success Rate Gauge */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">
              Overall Success Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col items-center justify-center h-48">
              <div className="relative w-40 h-40">
                <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                  {/* Background circle */}
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="10"
                    className="text-muted"
                  />
                  {/* Progress circle */}
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="10"
                    strokeDasharray={`${98.3 * 2.83} ${100 * 2.83}`}
                    className="text-green-500 transition-all duration-1000"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-3xl font-bold">98.3%</span>
                  <span className="text-xs text-muted-foreground">Success</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Crew Performance */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Crew Performance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {crewMetrics.map((crew, index) => (
              <div
                key={crew.name}
                className="space-y-2 animate-slide-up"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`h-3 w-3 rounded-full ${crew.color}`} />
                    <span className="font-medium">{crew.name}</span>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <span>{crew.tasks} tasks</span>
                    <span>{crew.success}% success</span>
                    <span>{crew.avgDuration}s avg</span>
                  </div>
                </div>
                <Progress value={crew.success} className="h-2" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
