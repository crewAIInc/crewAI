"use strict";

/**
 * Flow Visualization Interactive Script
 * Handles the interactive network visualization for CrewAI flows
 */

// ============================================================================
// Constants
// ============================================================================

const CONSTANTS = {
  NODE: {
    BASE_WIDTH: 200,
    BASE_HEIGHT: 60,
    BORDER_RADIUS: 20,
    TEXT_SIZE: 13,
    TEXT_PADDING: 8,
    TEXT_BG_RADIUS: 6,
    HOVER_SCALE: 1.04,
    PRESSED_SCALE: 0.98,
  },
  EDGE: {
    DEFAULT_WIDTH: 2,
    HIGHLIGHTED_WIDTH: 8,
    ANIMATION_DURATION: 300,
    DEFAULT_SHADOW_SIZE: 4,
    HIGHLIGHTED_SHADOW_SIZE: 20,
  },
  ANIMATION: {
    DURATION: 300,
    EASE_OUT_CUBIC: (t) => 1 - Math.pow(1 - t, 3),
  },
  NETWORK: {
    STABILIZATION_ITERATIONS: 200,
    NODE_DISTANCE: 180,
    SPRING_LENGTH: 150,
    LEVEL_SEPARATION: 180,
    NODE_SPACING: 220,
    TREE_SPACING: 250,
  },
  DRAWER: {
    WIDTH: 400,
    OFFSET_SCALE: 0.3,
  },
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Loads the vis-network library from CDN
 */
function loadVisCDN() {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js";
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

/**
 * Draws a rounded rectangle on a canvas context
 */
function drawRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

/**
 * Highlights Python code using Prism
 */
function highlightPython(code) {
  return Prism.highlight(code, Prism.languages.python, "python");
}

// ============================================================================
// Node Renderer
// ============================================================================

class NodeRenderer {
  constructor(nodes, networkManager) {
    this.nodes = nodes;
    this.networkManager = networkManager;
  }

  render({ ctx, id, x, y, state, style, label }) {
    const node = this.nodes.get(id);
    if (!node || !node.nodeStyle) return {};

    const scale = this.getNodeScale(id);
    const isActiveDrawer =
      this.networkManager.drawerManager?.activeNodeId === id;
    const nodeStyle = node.nodeStyle;
    const width = CONSTANTS.NODE.BASE_WIDTH * scale;
    const height = CONSTANTS.NODE.BASE_HEIGHT * scale;

    return {
      drawNode: () => {
        ctx.save();
        this.applyNodeOpacity(ctx, node);
        this.applyShadow(ctx, node, isActiveDrawer);
        this.drawNodeShape(
          ctx,
          x,
          y,
          width,
          height,
          scale,
          nodeStyle,
          isActiveDrawer,
        );
        this.drawNodeText(ctx, x, y, scale, nodeStyle);
        ctx.restore();
      },
      nodeDimensions: { width, height },
    };
  }

  getNodeScale(id) {
    if (this.networkManager.pressedNodeId === id) {
      return CONSTANTS.NODE.PRESSED_SCALE;
    } else if (this.networkManager.hoveredNodeId === id) {
      return CONSTANTS.NODE.HOVER_SCALE;
    }
    return 1.0;
  }

  applyNodeOpacity(ctx, node) {
    const nodeOpacity = node.opacity !== undefined ? node.opacity : 1.0;
    ctx.globalAlpha = nodeOpacity;
  }

  applyShadow(ctx, node, isActiveDrawer) {
    if (node.shadow && node.shadow.enabled) {
      ctx.shadowColor = node.shadow.color || "rgba(0,0,0,0.1)";
      ctx.shadowBlur = node.shadow.size || 8;
      ctx.shadowOffsetX = node.shadow.x || 0;
      ctx.shadowOffsetY = node.shadow.y || 0;
    } else if (isActiveDrawer) {
      ctx.shadowColor = "{{ CREWAI_ORANGE }}";
      ctx.shadowBlur = 20;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
    }
  }

  drawNodeShape(ctx, x, y, width, height, scale, nodeStyle, isActiveDrawer) {
    const radius = CONSTANTS.NODE.BORDER_RADIUS * scale;
    const rectX = x - width / 2;
    const rectY = y - height / 2;

    drawRoundedRect(ctx, rectX, rectY, width, height, radius);

    ctx.fillStyle = nodeStyle.bgColor;
    ctx.fill();

    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;

    ctx.strokeStyle = isActiveDrawer
      ? "{{ CREWAI_ORANGE }}"
      : nodeStyle.borderColor;
    ctx.lineWidth = nodeStyle.borderWidth * scale;
    ctx.stroke();
  }

  drawNodeText(ctx, x, y, scale, nodeStyle) {
    ctx.font = `500 ${CONSTANTS.NODE.TEXT_SIZE * scale}px 'JetBrains Mono', 'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const textMetrics = ctx.measureText(nodeStyle.name);
    const textWidth = textMetrics.width;
    const textHeight = CONSTANTS.NODE.TEXT_SIZE * scale;
    const textPadding = CONSTANTS.NODE.TEXT_PADDING * scale;
    const textBgRadius = CONSTANTS.NODE.TEXT_BG_RADIUS * scale;

    const textBgX = x - textWidth / 2 - textPadding;
    const textBgY = y - textHeight / 2 - textPadding / 2;
    const textBgWidth = textWidth + textPadding * 2;
    const textBgHeight = textHeight + textPadding;

    drawRoundedRect(
      ctx,
      textBgX,
      textBgY,
      textBgWidth,
      textBgHeight,
      textBgRadius,
    );

    ctx.fillStyle = "rgba(255, 255, 255, 0.2)";
    ctx.fill();

    ctx.fillStyle = nodeStyle.fontColor;
    ctx.fillText(nodeStyle.name, x, y);
  }
}

// ============================================================================
// Animation Manager
// ============================================================================

class AnimationManager {
  constructor() {
    this.animations = new Map();
  }

  animateEdgeWidth(
    edges,
    edgeId,
    targetWidth,
    duration = CONSTANTS.EDGE.ANIMATION_DURATION,
  ) {
    this.cancel(edgeId);

    const edge = edges.get(edgeId);
    if (!edge) return;

    const startWidth = edge.width || CONSTANTS.EDGE.DEFAULT_WIDTH;
    const startTime = performance.now();

    const animate = () => {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);
      const currentWidth = startWidth + (targetWidth - startWidth) * eased;

      edges.update({ id: edgeId, width: currentWidth });

      if (progress < 1) {
        const frameId = requestAnimationFrame(animate);
        this.animations.set(edgeId, frameId);
      } else {
        this.animations.delete(edgeId);
      }
    };

    animate();
  }

  cancel(id) {
    if (this.animations.has(id)) {
      cancelAnimationFrame(this.animations.get(id));
      this.animations.delete(id);
    }
  }

  cancelAll() {
    this.animations.forEach((frameId) => cancelAnimationFrame(frameId));
    this.animations.clear();
  }
}

// ============================================================================
// Triggered By Highlighter
// ============================================================================

class TriggeredByHighlighter {
  constructor(network, nodes, edges, highlightCanvas) {
    this.network = network;
    this.nodes = nodes;
    this.edges = edges;
    this.canvas = highlightCanvas;
    this.ctx = highlightCanvas.getContext("2d");

    this.highlightedNodes = [];
    this.highlightedEdges = [];
    this.activeDrawerNodeId = null;
    this.activeDrawerEdges = [];

    this.setupCanvas();
  }

  setupCanvas() {
    this.resizeCanvas();
    this.canvas.classList.remove("visible");
    window.addEventListener("resize", () => this.resizeCanvas());
  }

  resizeCanvas() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  setActiveDrawer(nodeId, edges) {
    this.activeDrawerNodeId = nodeId;
    this.activeDrawerEdges = edges;
  }

  highlightTriggeredByGroup(triggerNodeIds) {
    this.clear();

    if (!this.activeDrawerNodeId || !triggerNodeIds || triggerNodeIds.length === 0) {
      console.warn("TriggeredByHighlighter: Missing activeDrawerNodeId or triggerNodeIds");
      return;
    }

    const allEdges = this.edges.get();
    const pathNodes = new Set([this.activeDrawerNodeId]);
    const connectingEdges = [];
    const nodeData = '{{ nodeData }}';

    triggerNodeIds.forEach(triggerNodeId => {
      const directEdges = allEdges.filter(
        (edge) => edge.from === triggerNodeId && edge.to === this.activeDrawerNodeId
      );

      if (directEdges.length > 0) {
        directEdges.forEach(edge => {
          connectingEdges.push(edge);
          pathNodes.add(edge.from);
          pathNodes.add(edge.to);
        });
      } else {
        for (const [nodeName, nodeInfo] of Object.entries(nodeData)) {
          if (nodeInfo.router_paths && nodeInfo.router_paths.includes(triggerNodeId)) {
            const routerNode = nodeName;

            const routerEdges = allEdges.filter(
              (edge) => edge.from === routerNode && edge.dashes
            );

            for (const routerEdge of routerEdges) {
              if (routerEdge.to === this.activeDrawerNodeId) {
                connectingEdges.push(routerEdge);
                pathNodes.add(routerNode);
                pathNodes.add(this.activeDrawerNodeId);
                break;
              }

              const intermediateNode = routerEdge.to;
              const pathToActive = allEdges.filter(
                (edge) => edge.from === intermediateNode && edge.to === this.activeDrawerNodeId
              );

              if (pathToActive.length > 0) {
                connectingEdges.push(routerEdge);
                connectingEdges.push(...pathToActive);
                pathNodes.add(routerNode);
                pathNodes.add(intermediateNode);
                pathNodes.add(this.activeDrawerNodeId);
                break;
              }
            }

            if (connectingEdges.length > 0) break;
          }
        }
      }
    });

    if (connectingEdges.length === 0) {
      console.warn("TriggeredByHighlighter: No connecting edges found for group", { triggerNodeIds });
      return;
    }

    this.highlightedNodes = Array.from(pathNodes);
    this.highlightedEdges = connectingEdges.map((e) => e.id);

    this.animateNodeOpacity();
    this.animateEdgeStyles();
  }

  highlightAllRouterPaths() {
    this.clear();

    if (!this.activeDrawerNodeId) {
      console.warn("TriggeredByHighlighter: Missing activeDrawerNodeId");
      return;
    }

    const allEdges = this.edges.get();
    const nodeData = '{{ nodeData }}';
    const activeMetadata = nodeData[this.activeDrawerNodeId];

    const outgoingRouterEdges = allEdges.filter(
      (edge) => edge.from === this.activeDrawerNodeId && edge.dashes
    );

    let routerEdges = [];
    const pathNodes = new Set();

    if (outgoingRouterEdges.length > 0) {
      routerEdges = outgoingRouterEdges;
      pathNodes.add(this.activeDrawerNodeId);
      routerEdges.forEach(edge => {
        pathNodes.add(edge.to);
      });
    } else if (activeMetadata && activeMetadata.router_paths && activeMetadata.router_paths.length > 0) {
      activeMetadata.router_paths.forEach(pathName => {
        for (const [nodeName, nodeInfo] of Object.entries(nodeData)) {
          if (nodeInfo.router_paths && nodeInfo.router_paths.includes(pathName)) {
            const edgeFromRouter = allEdges.filter(
              (edge) => edge.from === nodeName && edge.to === this.activeDrawerNodeId && edge.dashes
            );

            if (edgeFromRouter.length > 0) {
              routerEdges.push(...edgeFromRouter);
              pathNodes.add(nodeName);
              pathNodes.add(this.activeDrawerNodeId);
            }
          }
        }
      });
    }

    if (routerEdges.length === 0) {
      console.warn("TriggeredByHighlighter: No router paths found for node", {
        activeDrawerNodeId: this.activeDrawerNodeId,
        outgoingEdges: outgoingRouterEdges.length,
        hasRouterPathsMetadata: !!activeMetadata?.router_paths,
      });
      return;
    }

    this.highlightedNodes = Array.from(pathNodes);
    this.highlightedEdges = routerEdges.map((e) => e.id);

    this.animateNodeOpacity();
    this.animateEdgeStyles();
  }

  highlightTriggeredBy(triggerNodeId) {
    this.clear();

    if (this.activeDrawerEdges && this.activeDrawerEdges.length > 0) {
      this.activeDrawerEdges.forEach((edgeId) => {
        this.edges.update({
          id: edgeId,
          width: CONSTANTS.EDGE.DEFAULT_WIDTH,
          opacity: 1.0,
        });
      });
      this.activeDrawerEdges = [];
    }

    if (!this.activeDrawerNodeId || !triggerNodeId) {
      console.warn(
        "TriggeredByHighlighter: Missing activeDrawerNodeId or triggerNodeId",
        {
          activeDrawerNodeId: this.activeDrawerNodeId,
          triggerNodeId: triggerNodeId,
        },
      );
      return;
    }

    const allEdges = this.edges.get();
    let connectingEdges = [];
    let actualTriggerNodeId = triggerNodeId;

    connectingEdges = allEdges.filter(
      (edge) =>
        edge.from === triggerNodeId && edge.to === this.activeDrawerNodeId,
    );

    if (connectingEdges.length === 0) {
      const incomingRouterEdges = allEdges.filter(
        (edge) => edge.to === this.activeDrawerNodeId && edge.dashes,
      );

      if (incomingRouterEdges.length > 0) {
        incomingRouterEdges.forEach((edge) => {
          connectingEdges.push(edge);
          actualTriggerNodeId = edge.from;
        });
      }
    }

    if (connectingEdges.length === 0) {
      const outgoingRouterEdges = allEdges.filter(
        (edge) => edge.from === this.activeDrawerNodeId && edge.dashes,
      );

      if (outgoingRouterEdges.length > 0) {
        const nodeData = '{{ nodeData }}';
        for (const [nodeName, nodeInfo] of Object.entries(nodeData)) {
          if (
            nodeInfo.trigger_methods &&
            nodeInfo.trigger_methods.includes(triggerNodeId)
          ) {
            const edgeToTarget = outgoingRouterEdges.find(
              (e) => e.to === nodeName,
            );
            if (edgeToTarget) {
              connectingEdges.push(edgeToTarget);
              actualTriggerNodeId = nodeName;
              break;
            }
          }
        }
      }
    }

    if (connectingEdges.length === 0) {
      const nodeData = '{{ nodeData }}';

      const activeMetadata = nodeData[this.activeDrawerNodeId];
      if (
        activeMetadata &&
        activeMetadata.trigger_methods &&
        activeMetadata.trigger_methods.includes(triggerNodeId)
      ) {
        for (const [nodeName, nodeInfo] of Object.entries(nodeData)) {
          if (
            nodeInfo.router_paths &&
            nodeInfo.router_paths.includes(triggerNodeId)
          ) {
            const routerNode = nodeName;

            const routerEdges = allEdges.filter(
              (edge) => edge.from === routerNode && edge.dashes,
            );

            for (const routerEdge of routerEdges) {
              const intermediateNode = routerEdge.to;

              if (intermediateNode === this.activeDrawerNodeId) {
                connectingEdges = [routerEdge];
                actualTriggerNodeId = routerNode;
                break;
              }

              const pathToActive = allEdges.filter(
                (edge) =>
                  edge.from === intermediateNode &&
                  edge.to === this.activeDrawerNodeId,
              );

              if (pathToActive.length > 0) {
                connectingEdges = [routerEdge, ...pathToActive];
                actualTriggerNodeId = routerNode;
                break;
              }
            }

            if (connectingEdges.length > 0) break;
          }
        }
      }
    }

    if (connectingEdges.length === 0) {
      const edgesWithLabel = allEdges.filter(
        (edge) =>
          edge.dashes &&
          edge.label === triggerNodeId &&
          edge.to === this.activeDrawerNodeId,
      );

      if (edgesWithLabel.length > 0) {
        connectingEdges = edgesWithLabel;
        const firstEdge = edgesWithLabel[0];
        actualTriggerNodeId = firstEdge.from;
      }
    }

    if (connectingEdges.length === 0) {
      console.warn("TriggeredByHighlighter: No connecting edges found", {
        triggerNodeId,
        activeDrawerNodeId: this.activeDrawerNodeId,
        allEdges: allEdges.length,
        edgeDetails: allEdges.map((e) => ({
          from: e.from,
          to: e.to,
          label: e.label,
          dashes: e.dashes,
        })),
      });
      return;
    }

    const pathNodes = new Set([actualTriggerNodeId, this.activeDrawerNodeId]);
    connectingEdges.forEach((edge) => {
      pathNodes.add(edge.from);
      pathNodes.add(edge.to);
    });

    this.highlightedNodes = Array.from(pathNodes);
    this.highlightedEdges = connectingEdges.map((e) => e.id);

    this.animateNodeOpacity();
    this.animateEdgeStyles();
  }

  animateNodeOpacity() {
    const allNodesList = this.nodes.get();
    const nodeAnimDuration = CONSTANTS.ANIMATION.DURATION;
    const nodeAnimStart = performance.now();

    const animate = () => {
      const elapsed = performance.now() - nodeAnimStart;
      const progress = Math.min(elapsed / nodeAnimDuration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);

      allNodesList.forEach((node) => {
        const currentOpacity = node.opacity !== undefined ? node.opacity : 1.0;
        const targetOpacity = this.highlightedNodes.includes(node.id)
          ? 1.0
          : 0.2;
        const newOpacity =
          currentOpacity + (targetOpacity - currentOpacity) * eased;

        this.nodes.update({
          id: node.id,
          opacity: newOpacity,
        });
      });

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();
  }

  animateEdgeStyles() {
    const edgeIds = this.edges.getIds();
    const edgeAnimDuration = CONSTANTS.ANIMATION.DURATION;
    const edgeAnimStart = performance.now();

    const animate = () => {
      const elapsed = performance.now() - edgeAnimStart;
      const progress = Math.min(elapsed / edgeAnimDuration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);

      edgeIds.forEach((edgeId) => {
        const edge = this.edges.get(edgeId);
        if (!edge) return;

        if (this.highlightedEdges.includes(edge.id)) {
          const currentWidth = edge.width || CONSTANTS.EDGE.DEFAULT_WIDTH;
          const targetWidth = CONSTANTS.EDGE.HIGHLIGHTED_WIDTH;
          const newWidth = currentWidth + (targetWidth - currentWidth) * eased;

          const currentShadowSize =
            edge.shadow?.size || CONSTANTS.EDGE.DEFAULT_SHADOW_SIZE;
          const targetShadowSize = CONSTANTS.EDGE.HIGHLIGHTED_SHADOW_SIZE;
          const newShadowSize =
            currentShadowSize + (targetShadowSize - currentShadowSize) * eased;

          const updateData = {
            id: edge.id,
            hidden: false,
            opacity: 1.0,
            width: newWidth,
            color: {
              color: "{{ CREWAI_ORANGE }}",
              highlight: "{{ CREWAI_ORANGE }}",
            },
            shadow: {
              enabled: true,
              color: "{{ CREWAI_ORANGE }}",
              size: newShadowSize,
              x: 0,
              y: 0,
            },
          };

          if (edge.dashes) {
            const scale = Math.sqrt(newWidth / CONSTANTS.EDGE.DEFAULT_WIDTH);
            updateData.dashes = [15 * scale, 10 * scale];
          }

          updateData.arrows = {
            to: {
              enabled: true,
              scaleFactor: 0.8,
              type: "triangle",
            },
          };

          updateData.color = {
            color: "{{ CREWAI_ORANGE }}",
            highlight: "{{ CREWAI_ORANGE }}",
            hover: "{{ CREWAI_ORANGE }}",
            inherit: "to",
          };

          this.edges.update(updateData);
        } else {
          const currentOpacity = edge.opacity !== undefined ? edge.opacity : 1.0;
          const targetOpacity = 0.25;
          const newOpacity = currentOpacity + (targetOpacity - currentOpacity) * eased;

          const currentWidth = edge.width !== undefined ? edge.width : CONSTANTS.EDGE.DEFAULT_WIDTH;
          const targetWidth = 1;
          const newWidth = currentWidth + (targetWidth - currentWidth) * eased;

          this.edges.update({
            id: edge.id,
            hidden: false,
            opacity: newOpacity,
            width: newWidth,
            color: {
              color: "rgba(128, 128, 128, 0.3)",
              highlight: "rgba(128, 128, 128, 0.3)",
            },
            shadow: {
              enabled: false,
            },
          });
        }
      });

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();
  }

  drawHighlightLayer() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    if (this.highlightedNodes.length === 0) return;

    this.highlightedNodes.forEach((nodeId) => {
      const nodePosition = this.network.getPositions([nodeId])[nodeId];
      if (!nodePosition) return;

      const canvasPos = this.network.canvasToDOM(nodePosition);
      const node = this.nodes.get(nodeId);
      if (!node || !node.nodeStyle) return;

      const nodeStyle = node.nodeStyle;
      const scale = 1.0;
      const width = CONSTANTS.NODE.BASE_WIDTH * scale;
      const height = CONSTANTS.NODE.BASE_HEIGHT * scale;

      this.ctx.save();

      this.ctx.shadowColor = "transparent";
      this.ctx.shadowBlur = 0;
      this.ctx.shadowOffsetX = 0;
      this.ctx.shadowOffsetY = 0;

      const radius = CONSTANTS.NODE.BORDER_RADIUS * scale;
      const rectX = canvasPos.x - width / 2;
      const rectY = canvasPos.y - height / 2;

      drawRoundedRect(this.ctx, rectX, rectY, width, height, radius);

      this.ctx.fillStyle = nodeStyle.bgColor;
      this.ctx.fill();

      this.ctx.shadowColor = "transparent";
      this.ctx.shadowBlur = 0;

      this.ctx.strokeStyle = "{{ CREWAI_ORANGE }}";
      this.ctx.lineWidth = nodeStyle.borderWidth * scale;
      this.ctx.stroke();

      this.ctx.fillStyle = nodeStyle.fontColor;
      this.ctx.font = `500 ${15 * scale}px Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
      this.ctx.textAlign = "center";
      this.ctx.textBaseline = "middle";
      this.ctx.fillText(nodeStyle.name, canvasPos.x, canvasPos.y);

      this.ctx.restore();
    });
  }

  clear() {
    const allNodesList = this.nodes.get();
    const nodeRestoreAnimStart = performance.now();
    const nodeRestoreAnimDuration = CONSTANTS.ANIMATION.DURATION;

    const animate = () => {
      const elapsed = performance.now() - nodeRestoreAnimStart;
      const progress = Math.min(elapsed / nodeRestoreAnimDuration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);

      allNodesList.forEach((node) => {
        const currentOpacity = node.opacity !== undefined ? node.opacity : 1.0;
        const targetOpacity = 1.0;
        const newOpacity =
          currentOpacity + (targetOpacity - currentOpacity) * eased;
        this.nodes.update({ id: node.id, opacity: newOpacity });
      });

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();

    const edgeIds = this.edges.getIds();
    const edgeRestoreAnimStart = performance.now();
    const edgeRestoreAnimDuration = CONSTANTS.ANIMATION.DURATION;

    const animateEdges = () => {
      const elapsed = performance.now() - edgeRestoreAnimStart;
      const progress = Math.min(elapsed / edgeRestoreAnimDuration, 1);
      const eased = CONSTANTS.ANIMATION.EASE_OUT_CUBIC(progress);

      edgeIds.forEach((edgeId) => {
        if (this.activeDrawerEdges.includes(edgeId)) {
          return;
        }

        const edge = this.edges.get(edgeId);
        if (!edge) return;

        const defaultColor =
          edge.dashes || edge.label === "AND"
            ? "{{ CREWAI_ORANGE }}"
            : getComputedStyle(document.documentElement).getPropertyValue('--edge-or-color').trim();
        const currentOpacity = edge.opacity !== undefined ? edge.opacity : 1.0;
        const currentWidth =
          edge.width !== undefined ? edge.width : CONSTANTS.EDGE.DEFAULT_WIDTH;
        const currentShadowSize =
          edge.shadow && edge.shadow.size !== undefined
            ? edge.shadow.size
            : CONSTANTS.EDGE.DEFAULT_SHADOW_SIZE;

        const targetOpacity = 1.0;
        const targetWidth = CONSTANTS.EDGE.DEFAULT_WIDTH;
        const targetShadowSize = CONSTANTS.EDGE.DEFAULT_SHADOW_SIZE;

        const newOpacity =
          currentOpacity + (targetOpacity - currentOpacity) * eased;
        const newWidth = currentWidth + (targetWidth - currentWidth) * eased;
        const newShadowSize =
          currentShadowSize + (targetShadowSize - currentShadowSize) * eased;

        const updateData = {
          id: edge.id,
          hidden: false,
          opacity: newOpacity,
          width: newWidth,
          color: {
            color: defaultColor,
            highlight: defaultColor,
            hover: defaultColor,
            inherit: false,
          },
          shadow: {
            enabled: true,
            color: "rgba(0,0,0,0.08)",
            size: newShadowSize,
            x: 1,
            y: 1,
          },
          font: {
            color: "transparent",
            background: "transparent",
          },
          arrows: {
            to: {
              enabled: true,
              scaleFactor: 0.8,
              type: "triangle",
            },
          },
        };

        if (edge.dashes) {
          const scale = Math.sqrt(newWidth / CONSTANTS.EDGE.DEFAULT_WIDTH);
          updateData.dashes = [15 * scale, 10 * scale];
        }

        this.edges.update(updateData);
      });

      if (progress < 1) {
        requestAnimationFrame(animateEdges);
      }
    };

    animateEdges();

    this.highlightedNodes = [];
    this.highlightedEdges = [];

    this.canvas.style.transition = "opacity 300ms ease-out";
    this.canvas.style.opacity = "0";
    setTimeout(() => {
      this.canvas.classList.remove("visible");
      this.canvas.style.opacity = "1";
      this.canvas.style.transition = "";
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.network.redraw();
    }, 300);
  }
}

// ============================================================================
// Drawer Manager
// ============================================================================

class DrawerManager {
  constructor(network, nodes, edges, animationManager, triggeredByHighlighter) {
    this.network = network;
    this.nodes = nodes;
    this.edges = edges;
    this.animationManager = animationManager;
    this.triggeredByHighlighter = triggeredByHighlighter;

    this.elements = {
      drawer: document.getElementById("drawer"),
      overlay: document.getElementById("drawer-overlay"),
      title: document.getElementById("drawer-node-name"),
      content: document.getElementById("drawer-content"),
      openIdeButton: document.getElementById("drawer-open-ide"),
      closeButton: document.getElementById("drawer-close"),
      navControls: document.querySelector(".nav-controls"),
    };

    this.activeNodeId = null;
    this.activeEdges = [];

    this.setupEventListeners();
  }

  setupEventListeners() {
    this.elements.overlay.addEventListener("click", () => this.close());
    this.elements.closeButton.addEventListener("click", () => this.close());

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        this.close();
      }
    });
  }

  open(nodeName, metadata) {
    this.elements.title.textContent = nodeName;
    this.setupIdeButton(metadata);
    this.renderContent(nodeName, metadata);
    this.animateOpen();
  }

  setupIdeButton(metadata) {
    if (metadata.source_file && metadata.source_start_line) {
      this.elements.openIdeButton.style.display = "flex";
      this.elements.openIdeButton.onclick = () => this.openInIDE(metadata);
    } else {
      this.elements.openIdeButton.style.display = "none";
    }
  }

  openInIDE(metadata) {
    const filePath = metadata.source_file;
    const lineNum = metadata.source_start_line;
    const detectedIDE = this.detectIDE();

    const ideUrls = {
      pycharm: `pycharm://open?file=${filePath}&line=${lineNum}`,
      vscode: `vscode://file/${filePath}:${lineNum}`,
      jetbrains: `jetbrains://open?file=${encodeURIComponent(filePath)}&line=${lineNum}`,
      auto: `pycharm://open?file=${filePath}&line=${lineNum}`,
    };

    const ideUrl = ideUrls[detectedIDE] || ideUrls.auto;

    const link = document.createElement("a");
    link.href = ideUrl;
    link.target = "_blank";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    const fallbackText = `${filePath}:${lineNum}`;
    navigator.clipboard.writeText(fallbackText).catch((err) => {
      console.error("Failed to copy:", err);
    });
  }

  detectIDE() {
    const savedIDE = localStorage.getItem("preferred_ide");
    if (savedIDE) return savedIDE;
    if (navigator.userAgent.includes("JetBrains")) return "jetbrains";
    return "auto";
  }

  renderContent(nodeName, metadata) {
    let content = "";

    content += this.renderMetadata(metadata);

    if (metadata.source_code) {
      content += this.renderSourceCode(metadata);
    }

    this.elements.content.innerHTML = content;
    this.attachContentEventListeners(nodeName);
  }

  renderTriggerCondition(metadata) {
    if (metadata.trigger_condition) {
      return this.renderConditionTree(metadata.trigger_condition);
    } else if (metadata.trigger_methods) {
      return `
        <ul class="drawer-list">
          ${metadata.trigger_methods.map((t) => `<li><span class="drawer-code-link" data-node-id="${t}">${t}</span></li>`).join("")}
        </ul>
      `;
    }
    return "";
  }

  renderConditionTree(condition, depth = 0) {
    if (typeof condition === "string") {
      return `<span class="drawer-code-link trigger-leaf" data-node-id="${condition}">${condition}</span>`;
    }

    if (condition.type === "AND" || condition.type === "OR") {
      const conditionType = condition.type;
      const color = conditionType === "AND" ? "{{ CREWAI_ORANGE }}" : "var(--text-secondary)";
      const bgColor = conditionType === "AND" ? "rgba(255,90,80,0.08)" : "rgba(102,102,102,0.06)";
      const hoverBg = conditionType === "AND" ? "rgba(255,90,80,0.15)" : "rgba(102,102,102,0.12)";

      const triggerIds = this.extractTriggerIds(condition);
      const triggerIdsJson = JSON.stringify(triggerIds).replace(/"/g, '&quot;');

      const children = condition.conditions.map(sub => this.renderConditionTree(sub, depth + 1)).join("");

      return `
        <div class="condition-group" data-trigger-group="${triggerIdsJson}" style="border-left: 2px solid ${color}; padding: 8px 0 8px 12px; margin: 4px 0; transition: background 0.2s ease; position: relative; border-radius: 4px;" onmouseover="this.style.background='${hoverBg}'" onmouseout="this.style.background='transparent'">
          <div class="condition-label" data-condition-label="${conditionType}" style="color: ${color}; font-size: 11px; font-weight: 600; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; background: ${bgColor}; padding: 3px 8px; border-radius: 3px; display: inline-block; cursor: pointer; user-select: none;">
            ${conditionType} <span style="opacity: 0.5; font-size: 9px;">▼</span>
          </div>
          <div class="condition-children" style="margin-top: 4px;">
            ${children}
          </div>
        </div>
      `;
    }

    return "";
  }

  extractTriggerIds(condition) {
    if (typeof condition === "string") {
      return [condition];
    }

    if (condition.type === "AND" || condition.type === "OR") {
      const ids = [];
      condition.conditions.forEach(sub => {
        ids.push(...this.extractTriggerIds(sub));
      });
      return ids;
    }

    return [];
  }

  renderMetadata(metadata) {
    let metadataContent = "";

    const nodeType = metadata.type || "unknown";
    const typeBadgeColor =
      nodeType === "start" || nodeType === "router"
        ? "{{ CREWAI_ORANGE }}"
        : "{{ DARK_GRAY }}";
    metadataContent += `
            <div class="drawer-section">
                <div class="drawer-section-title">Type</div>
                <span class="drawer-badge" style="background: ${typeBadgeColor}; color: white;">${nodeType}</span>
            </div>
        `;

    if (metadata.condition_type) {
      const conditionColors = {
        AND: { color: "{{ CREWAI_ORANGE }}", bg: "rgba(255,90,80,0.12)" },
        IF: { color: "{{ CREWAI_ORANGE }}", bg: "rgba(255,90,80,0.18)" },
        default: { color: "{{ GRAY }}", bg: "rgba(102,102,102,0.12)" },
      };

      const { color, bg } =
        conditionColors[metadata.condition_type] || conditionColors.default;

      metadataContent += `
                <div class="drawer-section">
                    <div class="drawer-section-title">Condition</div>
                    <span class="drawer-badge" style="background: ${bg}; color: ${color};">${metadata.condition_type}</span>
                </div>
            `;
    }

    if (metadata.trigger_condition || (metadata.trigger_methods && metadata.trigger_methods.length > 0)) {
      metadataContent += `
                <div class="drawer-section">
                    <div class="drawer-section-title">Triggered By</div>
                    ${this.renderTriggerCondition(metadata)}
                </div>
            `;
    }

    if (metadata.router_paths && metadata.router_paths.length > 0) {
      const routerPathsJson = JSON.stringify(metadata.router_paths).replace(/"/g, '&quot;');
      metadataContent += `
                <div class="drawer-section">
                    <div class="drawer-section-title router-paths-title" data-router-paths="${routerPathsJson}" style="cursor: pointer; display: inline-flex; align-items: center; gap: 4px;">
                        Router Paths <span style="opacity: 0.5; font-size: 9px;">▼</span>
                    </div>
                    <ul class="drawer-list">
                        ${metadata.router_paths.map((p) => `<li><span class="drawer-code-link" data-node-id="${p}" style="color: {{ CREWAI_ORANGE }}; border-color: rgba(255,90,80,0.3);">${p}</span></li>`).join("")}
                    </ul>
                </div>
            `;
    }

    return metadataContent
      ? `<div class="drawer-metadata-grid">${metadataContent}</div>`
      : "";
  }

  renderSourceCode(metadata) {
    let lines = metadata.source_lines || metadata.source_code.split("\n");
    if (metadata.source_lines) {
      lines = lines.map((line) => line.replace(/\n$/, ""));
    }

    let minIndent = Infinity;
    lines.forEach((line) => {
      if (line.trim().length > 0) {
        const match = line.match(/^\s*/);
        const indent = match ? match[0].length : 0;
        minIndent = Math.min(minIndent, indent);
      }
    });

    const dedentedLines = lines.map((line) => {
      return line.trim().length === 0 ? "" : line.substring(minIndent);
    });

    const startLine = metadata.source_start_line || 1;
    const codeToHighlight = dedentedLines.join("\n").trim();
    const highlightedCode = highlightPython(codeToHighlight);

    const highlightedLines = highlightedCode.split("\n");
    const numberedLines = highlightedLines
      .map((line, index) => {
        const lineNum = startLine + index;
        return `<div class="code-line"><span class="line-number">${lineNum}</span>${line}</div>`;
      })
      .join("");

    let classSection = "";
    if (metadata.class_signature) {
      const highlightedClass = highlightPython(metadata.class_signature);
      let highlightedClassSignature = highlightedClass;

      if (metadata.class_line_number) {
        highlightedClassSignature = `<span class="line-number">${metadata.class_line_number}</span>${highlightedClass}`;
      }

      classSection = `
                <div>
                    <div class="accordion-subheader accordion-title">Class</div>
                    <div class="drawer-section">
                        <div class="code-block-container">
                            <pre class="code-block language-python">${highlightedClassSignature}</pre>
                        </div>
                    </div>
                </div>
            `;
    }

    return `
            <div class="accordion-section expanded">
                <div class="accordion-header">
                    <div class="accordion-title">Source Code</div>
                    <svg class="accordion-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="4,6 8,10 12,6" stroke-linecap="round" stroke-linejoin="round" />
                    </svg>
                </div>
                <div class="accordion-content">
                    <div class="drawer-section">
                        <div class="code-block-container">
                            <button class="code-copy-button" data-code="${codeToHighlight.replace(/"/g, "&quot;")}">
                                <svg class="copy-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
                                    <rect x="5" y="5" width="9" height="9" rx="1.5" />
                                    <path d="M3 11V3a2 2 0 0 1 2-2h8" />
                                </svg>
                                <svg class="check-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                                    <polyline points="3,8 6,11 13,4" stroke-linecap="round" stroke-linejoin="round" />
                                </svg>
                            </button>
                            <pre class="code-block language-python">${numberedLines}</pre>
                        </div>
                    </div>
                    ${classSection}
                </div>
            </div>
        `;
  }

  attachContentEventListeners(nodeName) {
    const copyButtons =
      this.elements.content.querySelectorAll(".code-copy-button");
    copyButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const codeText = button.getAttribute("data-code");
        navigator.clipboard
          .writeText(codeText)
          .then(() => {
            button.classList.add("copied");
            setTimeout(() => button.classList.remove("copied"), 2000);
          })
          .catch((err) => console.error("Failed to copy:", err));
      });
    });

    const accordionHeaders =
      this.elements.content.querySelectorAll(".accordion-header");
    accordionHeaders.forEach((header) => {
      header.addEventListener("click", () => {
        const accordionSection = header.closest(".accordion-section");
        accordionSection.classList.toggle("expanded");
      });
    });

    const triggeredByLinks = this.elements.content.querySelectorAll(
      ".drawer-code-link[data-node-id]",
    );
    triggeredByLinks.forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const triggerNodeId = link.getAttribute("data-node-id");
        this.triggeredByHighlighter.highlightTriggeredBy(triggerNodeId);
      });
    });

    const conditionGroups = this.elements.content.querySelectorAll(
      ".condition-group[data-trigger-group]",
    );
    conditionGroups.forEach((group) => {
      group.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const triggerIds = JSON.parse(group.getAttribute("data-trigger-group"));
        this.triggeredByHighlighter.highlightTriggeredByGroup(triggerIds);
      });
    });

    const routerPathsTitle = this.elements.content.querySelector(
      ".router-paths-title[data-router-paths]",
    );
    if (routerPathsTitle) {
      routerPathsTitle.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.triggeredByHighlighter.highlightAllRouterPaths();
      });
    }
  }

  animateOpen() {
    this.elements.drawer.style.visibility = "visible";
    const wasAlreadyOpen = this.elements.drawer.classList.contains("open");

    requestAnimationFrame(() => {
      this.elements.drawer.classList.add("open");
      this.elements.overlay.classList.add("visible");
      this.elements.navControls.classList.add("drawer-open");

      if (!wasAlreadyOpen) {
        setTimeout(() => {
          const currentScale = this.network.getScale();
          const currentPosition = this.network.getViewPosition();
          const offsetX =
            (CONSTANTS.DRAWER.WIDTH * CONSTANTS.DRAWER.OFFSET_SCALE) /
            currentScale;

          this.network.moveTo({
            position: {
              x: currentPosition.x + offsetX,
              y: currentPosition.y,
            },
            scale: currentScale,
            animation: {
              duration: 300,
              easingFunction: "easeInOutQuad",
            },
          });
        }, 50);
      }
    });
  }

  close() {
    this.elements.drawer.classList.remove("open");
    this.elements.overlay.classList.remove("visible");
    this.elements.navControls.classList.remove("drawer-open");

    if (this.activeNodeId) {
      this.activeEdges.forEach((edgeId) => {
        this.animationManager.animateEdgeWidth(
          this.edges,
          edgeId,
          CONSTANTS.EDGE.DEFAULT_WIDTH,
          200,
        );
      });
      this.activeNodeId = null;
      this.activeEdges = [];
    }

    this.triggeredByHighlighter.clear();

    setTimeout(() => {
      this.network.fit({
        animation: {
          duration: 300,
          easingFunction: "easeInOutQuad",
        },
      });
    }, 50);

    setTimeout(() => {
      if (!this.elements.drawer.classList.contains("open")) {
        this.elements.drawer.style.visibility = "hidden";
      }
    }, 300);
  }

  setActiveNode(nodeId, connectedEdges) {
    this.activeNodeId = nodeId;
    this.activeEdges = connectedEdges;
  }
}

// ============================================================================
// Network Manager
// ============================================================================

class NetworkManager {
  constructor() {
    this.network = null;
    this.nodes = null;
    this.edges = null;
    this.animationManager = new AnimationManager();
    this.drawerManager = null;
    this.triggeredByHighlighter = null;

    this.hoveredNodeId = null;
    this.pressedNodeId = null;
    this.pressedEdges = [];
    this.isClicking = false;
  }

  async initialize() {
    try {
      await loadVisCDN();

      this.nodes = new vis.DataSet('{{ nodes_list_json }}');
      this.edges = new vis.DataSet('{{ edges_list_json }}');

      const container = document.getElementById("network");
      const options = this.createNetworkOptions();

      this.network = new vis.Network(
        container,
        { nodes: this.nodes, edges: this.edges },
        options,
      );

      this.triggeredByHighlighter = new TriggeredByHighlighter(
        this.network,
        this.nodes,
        this.edges,
        document.getElementById("highlight-canvas"),
      );

      this.drawerManager = new DrawerManager(
        this.network,
        this.nodes,
        this.edges,
        this.animationManager,
        this.triggeredByHighlighter,
      );

      this.setupEventListeners();
      this.setupControls();
      this.setupTheme();

      this.network.once("stabilizationIterationsDone", () => {
        this.network.fit();
      });
    } catch (error) {
      console.error("Failed to initialize network:", error);
    }
  }

  createNetworkOptions() {
    const nodeRenderer = new NodeRenderer(this.nodes, this);

    return {
      nodes: {
        shape: "custom",
        shadow: false,
        chosen: false,
        size: 30,
        ctxRenderer: (params) => nodeRenderer.render(params),
        scaling: {
          min: 1,
          max: 100,
        },
      },
      edges: {
        width: CONSTANTS.EDGE.DEFAULT_WIDTH,
        hoverWidth: 0,
        labelHighlightBold: false,
        shadow: false,
        smooth: {
          type: "cubicBezier",
          roundness: 0.5,
        },
        font: {
          size: 13,
          align: "middle",
          color: "transparent",
          face: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          strokeWidth: 0,
          background: "transparent",
          vadjust: 0,
        },
        arrows: {
          to: {
            enabled: true,
            scaleFactor: 0.8,
            type: "triangle",
          },
        },
        arrowStrikethrough: true,
        chosen: {
          edge: false,
          label: false,
        },
      },
      physics: {
        enabled: true,
        hierarchicalRepulsion: {
          nodeDistance: CONSTANTS.NETWORK.NODE_DISTANCE,
          centralGravity: 0.0,
          springLength: CONSTANTS.NETWORK.SPRING_LENGTH,
          springConstant: 0.01,
          damping: 0.09,
        },
        solver: "hierarchicalRepulsion",
        stabilization: {
          enabled: true,
          iterations: CONSTANTS.NETWORK.STABILIZATION_ITERATIONS,
          updateInterval: 25,
        },
      },
      layout: {
        hierarchical: {
          enabled: true,
          direction: "UD",
          sortMethod: "directed",
          levelSeparation: CONSTANTS.NETWORK.LEVEL_SEPARATION,
          nodeSpacing: CONSTANTS.NETWORK.NODE_SPACING,
          treeSpacing: CONSTANTS.NETWORK.TREE_SPACING,
        },
      },
      interaction: {
        hover: true,
        hoverConnectedEdges: false,
        navigationButtons: false,
        keyboard: true,
        selectConnectedEdges: false,
        multiselect: false,
      },
    };
  }

  setupEventListeners() {
    this.network.on("hoverNode", (params) => {
      this.hoveredNodeId = params.node;
      document.body.style.cursor = "pointer";
      this.network.redraw();
    });

    this.network.on("blurNode", (params) => {
      if (this.hoveredNodeId === params.node) {
        this.hoveredNodeId = null;
      }
      document.body.style.cursor = "default";
      this.network.redraw();
    });

    this.network.on("selectNode", (params) => {
      if (params.nodes.length > 0) {
        this.pressedNodeId = params.nodes[0];
        this.pressedEdges = this.network.getConnectedEdges(this.pressedNodeId);
        this.network.redraw();
      }
    });

    this.network.on("deselectNode", () => {
      if (this.pressedNodeId) {
        setTimeout(() => {
          if (this.isClicking) {
            this.isClicking = false;
            this.pressedNodeId = null;
            this.pressedEdges = [];
            return;
          }

          this.pressedEdges.forEach((edgeId) => {
            if (!this.drawerManager.activeEdges.includes(edgeId)) {
              this.animationManager.animateEdgeWidth(
                this.edges,
                edgeId,
                CONSTANTS.EDGE.DEFAULT_WIDTH,
                150,
              );
            }
          });

          this.pressedNodeId = null;
          this.pressedEdges = [];
          this.network.redraw();
        }, 10);
      }
    });

    this.network.on("click", (params) => {
      if (params.nodes.length > 0) {
        this.pressedNodeId = params.nodes[0];
        this.network.redraw();
        this.handleNodeClick(params.nodes[0]);
      } else if (params.edges.length === 0) {
        this.pressedNodeId = null;
        this.network.redraw();
        this.drawerManager.close();
      }
    });

    this.network.on("afterDrawing", () => {
      if (this.triggeredByHighlighter.canvas.classList.contains("visible")) {
        this.triggeredByHighlighter.drawHighlightLayer();
      }
    });
  }

  handleNodeClick(nodeId) {
    const node = this.nodes.get(nodeId);
    const nodeData = '{{ nodeData }}';
    const metadata = nodeData[nodeId];

    this.isClicking = true;

    if (
      this.drawerManager.activeNodeId &&
      this.drawerManager.activeNodeId !== nodeId
    ) {
      this.drawerManager.activeEdges.forEach((edgeId) => {
        this.animationManager.animateEdgeWidth(
          this.edges,
          edgeId,
          CONSTANTS.EDGE.DEFAULT_WIDTH,
          200,
        );
      });
      this.triggeredByHighlighter.clear();
    }

    const connectedEdges = this.network.getConnectedEdges(nodeId);
    this.drawerManager.setActiveNode(nodeId, connectedEdges);
    this.triggeredByHighlighter.setActiveDrawer(nodeId, connectedEdges);

    setTimeout(() => {
      connectedEdges.forEach((edgeId) => {
        this.animationManager.animateEdgeWidth(this.edges, edgeId, 5, 200);
      });
    }, 15);

    this.drawerManager.open(nodeId, metadata);
  }

  setupControls() {
    const ideSelector = document.getElementById("ide-selector");
    const savedIDE = localStorage.getItem("preferred_ide") || "auto";
    ideSelector.value = savedIDE;
    ideSelector.addEventListener("change", function () {
      localStorage.setItem("preferred_ide", this.value);
    });

    document.getElementById("zoom-in").addEventListener("click", () => {
      const scale = this.network.getScale();
      this.network.moveTo({
        scale: scale * 1.2,
        animation: { duration: 200, easingFunction: "easeInOutQuad" },
      });
    });

    document.getElementById("zoom-out").addEventListener("click", () => {
      const scale = this.network.getScale();
      this.network.moveTo({
        scale: scale * 0.8,
        animation: { duration: 200, easingFunction: "easeInOutQuad" },
      });
    });

    document.getElementById("fit").addEventListener("click", () => {
      this.network.fit({
        animation: { duration: 300, easingFunction: "easeInOutQuad" },
      });
    });

    this.setupExportControls();
  }

  setupExportControls() {
    document.getElementById("export-png").addEventListener("click", () => {
      const script = document.createElement("script");
      script.src =
        "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js";
      script.onload = () => {
        html2canvas(document.getElementById("network-container")).then(
          (canvas) => {
            const link = document.createElement("a");
            link.download = "flow_dag.png";
            link.href = canvas.toDataURL();
            link.click();
          },
        );
      };
      document.head.appendChild(script);
    });

    document.getElementById("export-pdf").addEventListener("click", () => {
      const script1 = document.createElement("script");
      script1.src =
        "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js";
      script1.onload = () => {
        const script2 = document.createElement("script");
        script2.src =
          "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";
        script2.onload = () => {
          html2canvas(document.getElementById("network-container")).then(
            (canvas) => {
              const imgData = canvas.toDataURL("image/png");
              const { jsPDF } = window.jspdf;
              const pdf = new jsPDF({
                orientation:
                  canvas.width > canvas.height ? "landscape" : "portrait",
                unit: "px",
                format: [canvas.width, canvas.height],
              });
              pdf.addImage(imgData, "PNG", 0, 0, canvas.width, canvas.height);
              pdf.save("flow_dag.pdf");
            },
          );
        };
        document.head.appendChild(script2);
      };
      document.head.appendChild(script1);
    });

    document.getElementById("export-json").addEventListener("click", () => {
      const dagData = '{{ dagData }}';
      const dataStr = JSON.stringify(dagData, null, 2);
      const blob = new Blob([dataStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.download = "flow_dag.json";
      link.href = url;
      link.click();
      URL.revokeObjectURL(url);
    });
  }

  setupTheme() {
    const themeToggle = document.getElementById("theme-toggle");
    const htmlElement = document.documentElement;

    const updateEdgeColors = () => {
      const orEdgeColor = getComputedStyle(document.documentElement).getPropertyValue('--edge-or-color').trim();

      this.edges.forEach((edge) => {
        let edgeColor;
        if (edge.dashes || edge.label === "AND") {
          edgeColor = "{{ CREWAI_ORANGE }}";
        } else {
          edgeColor = orEdgeColor;
        }

        const updateData = {
          id: edge.id,
          color: {
            color: edgeColor,
            highlight: edgeColor,
          },
          font: {
            color: "transparent",
            background: "transparent",
          },
        };

        this.edges.update(updateData);
      });

      this.network.redraw();
    };

    const savedTheme = localStorage.getItem("theme") || "light";
    if (savedTheme === "dark") {
      htmlElement.setAttribute("data-theme", "dark");
      themeToggle.textContent = "☀️";
      themeToggle.title = "Toggle Light Mode";
      setTimeout(updateEdgeColors, 0);
    } else {
      setTimeout(updateEdgeColors, 0);
    }

    themeToggle.addEventListener("click", () => {
      const currentTheme = htmlElement.getAttribute("data-theme");
      const newTheme = currentTheme === "dark" ? "light" : "dark";

      if (newTheme === "dark") {
        htmlElement.setAttribute("data-theme", "dark");
        themeToggle.textContent = "☀️";
        themeToggle.title = "Toggle Light Mode";
      } else {
        htmlElement.removeAttribute("data-theme");
        themeToggle.textContent = "🌙";
        themeToggle.title = "Toggle Dark Mode";
      }

      localStorage.setItem("theme", newTheme);
      setTimeout(updateEdgeColors, 50);
    });
  }
}

// ============================================================================
// Application Entry Point
// ============================================================================

(async () => {
  const networkManager = new NetworkManager();
  await networkManager.initialize();
})();
