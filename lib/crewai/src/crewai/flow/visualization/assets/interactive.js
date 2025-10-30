function loadVisCDN() {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js';
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

(async () => {

    try {
        await loadVisCDN();
        const nodes = new vis.DataSet('{{ nodes_list_json }}');
        const edges = new vis.DataSet('{{ edges_list_json }}');

        const container = document.getElementById('network');
        const data = {
            nodes: nodes,
            edges: edges
        };

        const options = {
            nodes: {
                shape: 'custom',
                shadow: false,
                chosen: false,
                size: 30,
                ctxRenderer: function ({ctx, id, x, y, state: {selected, hover}, style, label}) {
                    const node = nodes.get(id);
                    if (!node || !node.nodeStyle) return {};

                    const nodeStyle = node.nodeStyle;
                    const baseWidth = 200;
                    const baseHeight = 60;

                    let scale = 1.0;
                    if (pressedNodeId === id) {
                        scale = 0.98;
                    } else if (hoveredNodeId === id) {
                        scale = 1.04;
                    }

                    const isActiveDrawer = activeDrawerNodeId === id;

                    const width = baseWidth * scale;
                    const height = baseHeight * scale;

                    return {
                        drawNode() {
                            ctx.save();

                            const nodeOpacity = node.opacity !== undefined ? node.opacity : 1.0;
                            ctx.globalAlpha = nodeOpacity;

                            if (isActiveDrawer) {
                                ctx.shadowColor = '{{ CREWAI_ORANGE }}';
                                ctx.shadowBlur = 20;
                                ctx.shadowOffsetX = 0;
                                ctx.shadowOffsetY = 0;
                            } else {
                                ctx.shadowColor = 'rgba(0,0,0,0.1)';
                                ctx.shadowBlur = 8;
                                ctx.shadowOffsetX = 2;
                                ctx.shadowOffsetY = 2;
                            }

                            const radius = 20 * scale;
                            const rectX = x - width / 2;
                            const rectY = y - height / 2;

                            ctx.beginPath();
                            ctx.moveTo(rectX + radius, rectY);
                            ctx.lineTo(rectX + width - radius, rectY);
                            ctx.quadraticCurveTo(rectX + width, rectY, rectX + width, rectY + radius);
                            ctx.lineTo(rectX + width, rectY + height - radius);
                            ctx.quadraticCurveTo(rectX + width, rectY + height, rectX + width - radius, rectY + height);
                            ctx.lineTo(rectX + radius, rectY + height);
                            ctx.quadraticCurveTo(rectX, rectY + height, rectX, rectY + height - radius);
                            ctx.lineTo(rectX, rectY + radius);
                            ctx.quadraticCurveTo(rectX, rectY, rectX + radius, rectY);
                            ctx.closePath();

                            ctx.fillStyle = nodeStyle.bgColor;
                            ctx.fill();

                            ctx.shadowColor = 'transparent';
                            ctx.shadowBlur = 0;

                            const borderWidth = isActiveDrawer ? nodeStyle.borderWidth * 2 : nodeStyle.borderWidth;
                            ctx.strokeStyle = isActiveDrawer ? '{{ CREWAI_ORANGE }}' : nodeStyle.borderColor;
                            ctx.lineWidth = borderWidth * scale;
                            ctx.stroke();

                            ctx.font = `500 ${13 * scale}px 'JetBrains Mono', 'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace`;
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';

                            const textMetrics = ctx.measureText(nodeStyle.name);
                            const textWidth = textMetrics.width;
                            const textHeight = 13 * scale;
                            const textPadding = 8 * scale;
                            const textBgRadius = 6 * scale;

                            const textBgX = x - textWidth / 2 - textPadding;
                            const textBgY = y - textHeight / 2 - textPadding / 2;
                            const textBgWidth = textWidth + textPadding * 2;
                            const textBgHeight = textHeight + textPadding;

                            ctx.beginPath();
                            ctx.moveTo(textBgX + textBgRadius, textBgY);
                            ctx.lineTo(textBgX + textBgWidth - textBgRadius, textBgY);
                            ctx.quadraticCurveTo(textBgX + textBgWidth, textBgY, textBgX + textBgWidth, textBgY + textBgRadius);
                            ctx.lineTo(textBgX + textBgWidth, textBgY + textBgHeight - textBgRadius);
                            ctx.quadraticCurveTo(textBgX + textBgWidth, textBgY + textBgHeight, textBgX + textBgWidth - textBgRadius, textBgY + textBgHeight);
                            ctx.lineTo(textBgX + textBgRadius, textBgY + textBgHeight);
                            ctx.quadraticCurveTo(textBgX, textBgY + textBgHeight, textBgX, textBgY + textBgHeight - textBgRadius);
                            ctx.lineTo(textBgX, textBgY + textBgRadius);
                            ctx.quadraticCurveTo(textBgX, textBgY, textBgX + textBgRadius, textBgY);
                            ctx.closePath();

                            ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
                            ctx.fill();

                            ctx.fillStyle = nodeStyle.fontColor;
                            ctx.fillText(nodeStyle.name, x, y);

                            ctx.restore();
                        },
                        nodeDimensions: {width, height}
                    };
                },
                scaling: {
                    min: 1,
                    max: 100
                }
            },
            edges: {
                width: 2,
                hoverWidth: 0,
                labelHighlightBold: false,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.08)',
                    size: 4,
                    x: 1,
                    y: 1
                },
                smooth: {
                    type: 'cubicBezier',
                    roundness: 0.5
                },
                font: {
                    size: 13,
                    align: 'middle',
                    color: 'transparent',
                    face: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                    strokeWidth: 0,
                    background: 'transparent',
                    vadjust: 0,
                    bold: {
                        face: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                        mod: 'bold',
                        vadjust: 0
                    }
                },
                arrows: {
                    to: {
                        enabled: true,
                        scaleFactor: 0.8,
                        type: 'triangle'
                    }
                },
                arrowStrikethrough: true,
                chosen: {
                    edge: false,
                    label: false
                }
            },
            physics: {
                enabled: true,
                hierarchicalRepulsion: {
                    nodeDistance: 180,
                    centralGravity: 0.0,
                    springLength: 150,
                    springConstant: 0.01,
                    damping: 0.09
                },
                solver: 'hierarchicalRepulsion',
                stabilization: {
                    enabled: true,
                    iterations: 200,
                    updateInterval: 25
                }
            },
            layout: {
                hierarchical: {
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 180,
                    nodeSpacing: 220,
                    treeSpacing: 250
                }
            },
            interaction: {
                hover: true,
                hoverConnectedEdges: false,
                navigationButtons: false,
                keyboard: true,
                selectConnectedEdges: false,
                multiselect: false
            }
        };

        const network = new vis.Network(container, data, options);

        const ideSelector = document.getElementById('ide-selector');
        const savedIDE = localStorage.getItem('preferred_ide') || 'auto';
        ideSelector.value = savedIDE;
        ideSelector.addEventListener('change', function () {
            localStorage.setItem('preferred_ide', this.value);
            console.log('IDE preference saved:', this.value);
        });

        const highlightCanvas = document.getElementById('highlight-canvas');
        const highlightCtx = highlightCanvas.getContext('2d');

        function resizeHighlightCanvas() {
            highlightCanvas.width = window.innerWidth;
            highlightCanvas.height = window.innerHeight;
        }

        resizeHighlightCanvas();
        window.addEventListener('resize', resizeHighlightCanvas);

        let highlightedNodes = [];
        let highlightedEdges = [];

        function drawHighlightLayer() {
            highlightCtx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);

            if (highlightedNodes.length === 0) return;

            highlightedNodes.forEach(function (nodeId) {
                const nodePosition = network.getPositions([nodeId])[nodeId];
                if (!nodePosition) return;

                const canvasPos = network.canvasToDOM(nodePosition);
                const node = nodes.get(nodeId);
                if (!node || !node.nodeStyle) return;

                const nodeStyle = node.nodeStyle;
                const baseWidth = 200;
                const baseHeight = 60;
                const scale = 1.0;
                const width = baseWidth * scale;
                const height = baseHeight * scale;

                highlightCtx.save();

                highlightCtx.shadowColor = '{{ CREWAI_ORANGE }}';
                highlightCtx.shadowBlur = 20;
                highlightCtx.shadowOffsetX = 0;
                highlightCtx.shadowOffsetY = 0;

                const radius = 20 * scale;
                const rectX = canvasPos.x - width / 2;
                const rectY = canvasPos.y - height / 2;

                highlightCtx.beginPath();
                highlightCtx.moveTo(rectX + radius, rectY);
                highlightCtx.lineTo(rectX + width - radius, rectY);
                highlightCtx.quadraticCurveTo(rectX + width, rectY, rectX + width, rectY + radius);
                highlightCtx.lineTo(rectX + width, rectY + height - radius);
                highlightCtx.quadraticCurveTo(rectX + width, rectY + height, rectX + width - radius, rectY + height);
                highlightCtx.lineTo(rectX + radius, rectY + height);
                highlightCtx.quadraticCurveTo(rectX, rectY + height, rectX, rectY + height - radius);
                highlightCtx.lineTo(rectX, rectY + radius);
                highlightCtx.quadraticCurveTo(rectX, rectY, rectX + radius, rectY);
                highlightCtx.closePath();

                highlightCtx.fillStyle = nodeStyle.bgColor;
                highlightCtx.fill();

                highlightCtx.shadowColor = 'transparent';
                highlightCtx.shadowBlur = 0;

                highlightCtx.strokeStyle = '{{ CREWAI_ORANGE }}';
                highlightCtx.lineWidth = nodeStyle.borderWidth * 2 * scale;
                highlightCtx.stroke();

                highlightCtx.fillStyle = nodeStyle.fontColor;
                highlightCtx.font = `500 ${15 * scale}px Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
                highlightCtx.textAlign = 'center';
                highlightCtx.textBaseline = 'middle';
                highlightCtx.fillText(nodeStyle.name, canvasPos.x, canvasPos.y);

                highlightCtx.restore();
            });
        }

        function highlightTriggeredBy(triggerNodeId) {
            clearTriggeredByHighlight();

            if (activeDrawerEdges && activeDrawerEdges.length > 0) {
                activeDrawerEdges.forEach(function (edgeId) {
                    edges.update({
                        id: edgeId,
                        width: 2,
                        opacity: 1.0
                    });
                });
                activeDrawerEdges = [];
            }

            if (!activeDrawerNodeId || !triggerNodeId) return;

            const allEdges = edges.get();
            let connectingEdges = [];
            let actualTriggerNodeId = triggerNodeId;

            connectingEdges = allEdges.filter(edge =>
                edge.from === triggerNodeId && edge.to === activeDrawerNodeId
            );

            if (connectingEdges.length === 0) {
                const incomingRouterEdges = allEdges.filter(edge =>
                    edge.to === activeDrawerNodeId && edge.dashes
                );

                if (incomingRouterEdges.length > 0) {
                    incomingRouterEdges.forEach(function (edge) {
                        connectingEdges.push(edge);
                        actualTriggerNodeId = edge.from;
                    });
                } else {
                    const outgoingRouterEdges = allEdges.filter(edge =>
                        edge.from === activeDrawerNodeId && edge.dashes
                    );

                    const nodeData = '{{ nodeData }}';
                    for (const [nodeName, nodeInfo] of Object.entries(nodeData)) {
                        if (nodeInfo.trigger_methods && nodeInfo.trigger_methods.includes(triggerNodeId)) {
                            const edgeToTarget = outgoingRouterEdges.find(e => e.to === nodeName);
                            if (edgeToTarget) {
                                connectingEdges.push(edgeToTarget);
                                actualTriggerNodeId = nodeName;
                                break;
                            }
                        }
                    }
                }
            }

            if (connectingEdges.length === 0) return;

            highlightedNodes = [actualTriggerNodeId, activeDrawerNodeId];
            highlightedEdges = connectingEdges.map(e => e.id);

            const allNodesList = nodes.get();
            const nodeAnimDuration = 300;
            const nodeAnimStart = performance.now();

            function animateNodeOpacity() {
                const elapsed = performance.now() - nodeAnimStart;
                const progress = Math.min(elapsed / nodeAnimDuration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);

                allNodesList.forEach(function (node) {
                    const currentOpacity = node.opacity !== undefined ? node.opacity : 1.0;
                    const targetOpacity = highlightedNodes.includes(node.id) ? 1.0 : 0.2;
                    const newOpacity = currentOpacity + (targetOpacity - currentOpacity) * eased;

                    nodes.update({
                        id: node.id,
                        opacity: newOpacity
                    });
                });

                if (progress < 1) {
                    requestAnimationFrame(animateNodeOpacity);
                }
            }

            animateNodeOpacity();

            const allEdgesList = edges.get();
            const edgeAnimDuration = 300;
            const edgeAnimStart = performance.now();

            function animateEdgeStyles() {
                const elapsed = performance.now() - edgeAnimStart;
                const progress = Math.min(elapsed / edgeAnimDuration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);

                allEdgesList.forEach(function (edge) {
                    if (highlightedEdges.includes(edge.id)) {
                        const currentWidth = edge.width || 2;
                        const targetWidth = 8;
                        const newWidth = currentWidth + (targetWidth - currentWidth) * eased;

                        const currentShadowSize = edge.shadow?.size || 4;
                        const targetShadowSize = 20;
                        const newShadowSize = currentShadowSize + (targetShadowSize - currentShadowSize) * eased;

                        const updateData = {
                            id: edge.id,
                            hidden: false,
                            opacity: 1.0,
                            width: newWidth,
                            color: {
                                color: '{{ CREWAI_ORANGE }}',
                                highlight: '{{ CREWAI_ORANGE }}'
                            },
                            shadow: {
                                enabled: true,
                                color: '{{ CREWAI_ORANGE }}',
                                size: newShadowSize,
                                x: 0,
                                y: 0
                            }
                        };

                        if (edge.dashes) {
                            const scale = Math.sqrt(newWidth / 2);
                            updateData.dashes = [15 * scale, 10 * scale];
                        }

                        updateData.arrows = {
                            to: {
                                enabled: true,
                                scaleFactor: 0.8,
                                type: 'triangle'
                            }
                        };

                        updateData.color = {
                            color: '{{ CREWAI_ORANGE }}',
                            highlight: '{{ CREWAI_ORANGE }}',
                            hover: '{{ CREWAI_ORANGE }}',
                            inherit: 'to'
                        };

                        edges.update(updateData);
                    } else {
                        edges.update({
                            id: edge.id,
                            hidden: false,
                            opacity: 1.0,
                            width: 1,
                            color: {
                                color: 'transparent',
                                highlight: 'transparent'
                            },
                            shadow: {
                                enabled: false
                            },
                            font: {
                                color: 'transparent',
                                background: 'transparent'
                            }
                        });
                    }
                });

                if (progress < 1) {
                    requestAnimationFrame(animateEdgeStyles);
                }
            }

            animateEdgeStyles();

            highlightCanvas.classList.add('visible');

            setTimeout(function () {
                drawHighlightLayer();
            }, 50);
        }

        function clearTriggeredByHighlight() {
            const allNodesList = nodes.get();
            const nodeRestoreAnimStart = performance.now();
            const nodeRestoreAnimDuration = 300;

            function animateNodeRestore() {
                const elapsed = performance.now() - nodeRestoreAnimStart;
                const progress = Math.min(elapsed / nodeRestoreAnimDuration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);

                allNodesList.forEach(function (node) {
                    const currentOpacity = node.opacity !== undefined ? node.opacity : 1.0;
                    const targetOpacity = 1.0;
                    const newOpacity = currentOpacity + (targetOpacity - currentOpacity) * eased;
                    nodes.update({id: node.id, opacity: newOpacity});
                });

                if (progress < 1) {
                    requestAnimationFrame(animateNodeRestore);
                }
            }

            animateNodeRestore();

            const allEdgesList = edges.get();
            const edgeRestoreAnimStart = performance.now();
            const edgeRestoreAnimDuration = 300;

            function animateEdgeRestore() {
                const elapsed = performance.now() - edgeRestoreAnimStart;
                const progress = Math.min(elapsed / edgeRestoreAnimDuration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);

                allEdgesList.forEach(function (edge) {
                    if (activeDrawerEdges.includes(edge.id)) {
                        return;
                    }

                    const defaultColor = edge.dashes || edge.label === 'AND' ? '{{ CREWAI_ORANGE }}' : '{{ GRAY }}';
                    const currentOpacity = edge.opacity !== undefined ? edge.opacity : 1.0;
                    const currentWidth = edge.width !== undefined ? edge.width : 2;
                    const currentShadowSize = edge.shadow && edge.shadow.size !== undefined ? edge.shadow.size : 4;

                    const targetOpacity = 1.0;
                    const targetWidth = 2;
                    const targetShadowSize = 4;

                    const newOpacity = currentOpacity + (targetOpacity - currentOpacity) * eased;
                    const newWidth = currentWidth + (targetWidth - currentWidth) * eased;
                    const newShadowSize = currentShadowSize + (targetShadowSize - currentShadowSize) * eased;

                    const updateData = {
                        id: edge.id,
                        hidden: false,
                        opacity: newOpacity,
                        width: newWidth,
                        color: {
                            color: defaultColor,
                            highlight: defaultColor
                        },
                        shadow: {
                            enabled: true,
                            color: 'rgba(0,0,0,0.08)',
                            size: newShadowSize,
                            x: 1,
                            y: 1
                        },
                        font: {
                            color: 'transparent',
                            background: 'transparent'
                        }
                    };

                    if (edge.dashes) {
                        const scale = Math.sqrt(newWidth / 2);
                        updateData.dashes = [15 * scale, 10 * scale];
                    }

                    edges.update(updateData);
                });

                if (progress < 1) {
                    requestAnimationFrame(animateEdgeRestore);
                }
            }

            animateEdgeRestore();

            highlightedNodes = [];
            highlightedEdges = [];

            highlightCanvas.style.transition = 'opacity 300ms ease-out';
            highlightCanvas.style.opacity = '0';
            setTimeout(function () {
                highlightCanvas.classList.remove('visible');
                highlightCanvas.style.opacity = '1';
                highlightCanvas.style.transition = '';
                highlightCtx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);
            }, 300);
        }

        network.on('afterDrawing', function () {
            if (highlightCanvas.classList.contains('visible')) {
                drawHighlightLayer();
            }
        });

        let hoveredNodeId = null;
        let pressedNodeId = null;
        let isClicking = false;
        let activeDrawerNodeId = null;
        let activeDrawerEdges = [];

        const edgeAnimations = {};

        function animateEdgeWidth(edgeId, targetWidth, duration) {
            if (edgeAnimations[edgeId]) {
                cancelAnimationFrame(edgeAnimations[edgeId].frameId);
            }

            const edge = edges.get(edgeId);
            if (!edge) return;

            const startWidth = edge.width || 2;
            const startTime = performance.now();

            function animate() {
                const currentTime = performance.now();
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                const currentWidth = startWidth + (targetWidth - startWidth) * eased;

                edges.update({
                    id: edgeId,
                    width: currentWidth
                });

                if (progress < 1) {
                    edgeAnimations[edgeId] = {
                        frameId: requestAnimationFrame(animate)
                    };
                } else {
                    delete edgeAnimations[edgeId];
                }
            }

            animate();
        }

        network.on('hoverNode', function (params) {
            const nodeId = params.node;
            hoveredNodeId = nodeId;
            document.body.style.cursor = 'pointer';
            network.redraw();
        });

        network.on('blurNode', function (params) {
            const nodeId = params.node;
            if (hoveredNodeId === nodeId) {
                hoveredNodeId = null;
            }
            document.body.style.cursor = 'default';
            network.redraw();
        });

        let pressedEdges = [];

        network.on('selectNode', function (params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                pressedNodeId = nodeId;

                const connectedEdges = network.getConnectedEdges(nodeId);
                pressedEdges = connectedEdges;

                network.redraw();
            }
        });

        network.on('deselectNode', function (params) {
            if (pressedNodeId) {
                const nodeId = pressedNodeId;

                setTimeout(function () {
                    if (isClicking) {
                        isClicking = false;
                        pressedNodeId = null;
                        pressedEdges = [];
                        return;
                    }

                    pressedNodeId = null;

                    pressedEdges.forEach(function (edgeId) {
                        if (!activeDrawerEdges.includes(edgeId)) {
                            animateEdgeWidth(edgeId, 2, 150);
                        }
                    });
                    pressedEdges = [];
                    network.redraw();
                }, 10);
            }
        });
        let highlightedNodeId = null;
        let highlightedSourceNodeId = null;
        let highlightedEdgeId = null;
        let highlightTimeout = null;
        let originalNodeData = null;
        let originalSourceNodeData = null;
        let originalEdgeData = null;
        let isAnimating = false;

        function clearHighlights() {
            console.log('clearHighlights called', highlightedNodeId, highlightedEdgeId);

            isAnimating = false;

            if (highlightTimeout) {
                clearTimeout(highlightTimeout);
                highlightTimeout = null;
            }

            // Restore all node opacities if we have the original values
            if (originalNodeData && originalNodeData.originalOpacities) {
                originalNodeData.originalOpacities.forEach((opacity, nodeId) => {
                    nodes.update({
                        id: nodeId,
                        opacity: opacity
                    });
                });
            }

            if (highlightedNodeId) {
                console.log('Clearing node highlight:', highlightedNodeId);
                if (originalNodeData && originalNodeData.shadow) {
                    nodes.update({
                        id: highlightedNodeId,
                        shadow: originalNodeData.shadow
                    });
                } else {
                    // Reset to default shadow
                    nodes.update({
                        id: highlightedNodeId,
                        shadow: {
                            enabled: true,
                            color: 'rgba(0,0,0,0.1)',
                            size: 8,
                            x: 2,
                            y: 2
                        }
                    });
                }
                highlightedNodeId = null;
                originalNodeData = null;
            }

            if (highlightedEdgeId) {
                console.log('Clearing edge highlight:', highlightedEdgeId);
                if (originalEdgeData && originalEdgeData.shadow) {
                    edges.update({
                        id: highlightedEdgeId,
                        shadow: originalEdgeData.shadow
                    });
                } else {
                    // Reset to default shadow
                    edges.update({
                        id: highlightedEdgeId,
                        shadow: {
                            enabled: true,
                            color: 'rgba(0,0,0,0.08)',
                            size: 4,
                            x: 1,
                            y: 1
                        }
                    });
                }
                highlightedEdgeId = null;
                originalEdgeData = null;
            }
        }


        function highlightPython(code) {
            return Prism.highlight(code, Prism.languages.python, 'python');
        }

        function highlightJson(jsonString) {
            let escaped = jsonString
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');

            let result = escaped
                .replace(/(&quot;)([^&]+?)(&quot;)(\\s*)(:)/g, '<span class="json-key">$1$2$3</span>$4<span class="json-colon">$5</span>')
                .replace(/(:)(\\s*)(&quot;)([^&]*?)(&quot;)/g, '<span class="json-colon">$1</span>$2<span class="string">$3$4$5</span>')
                .replace(/(:)(\\s*)([-+]?\\d+\\.?\\d*)/g, '<span class="json-colon">$1</span>$2<span class="number">$3</span>')
                .replace(/:\\s*(true|false)\\b/g, '<span class="json-colon">:</span> <span class="keyword">$1</span>')
                .replace(/:\\s*null\\b/g, '<span class="json-colon">:</span> <span class="keyword">null</span>')
                .replace(/([{\\[\\]}])/g, '<span class="json-bracket">$1</span>')
                .replace(/,/g, '<span class="json-bracket">,</span>');

            return result;
        }

        network.on('click', function (params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                const nodeData = '{{ nodeData }}';
                const metadata = nodeData[nodeId];

                isClicking = true;

                clearTriggeredByHighlight();
                if (activeDrawerNodeId && activeDrawerNodeId !== nodeId) {
                    activeDrawerEdges.forEach(function (edgeId) {
                        animateEdgeWidth(edgeId, 2, 200);
                    });
                }

                activeDrawerNodeId = nodeId;
                const connectedEdges = network.getConnectedEdges(nodeId);
                activeDrawerEdges = connectedEdges;

                setTimeout(function () {
                    activeDrawerEdges.forEach(function (edgeId) {
                        animateEdgeWidth(edgeId, 5, 200);
                    });

                    network.redraw();
                }, 15);

                openDrawer(nodeId, metadata);
                clearHighlights();
            } else if (params.edges.length === 0) {
                clearHighlights();
                closeDrawer();
            }
        });

        function openDrawer(nodeName, metadata) {
            const drawer = document.getElementById('drawer');
            const overlay = document.getElementById('drawer-overlay');
            const drawerTitle = document.getElementById('drawer-node-name');
            const drawerContent = document.getElementById('drawer-content');
            const openIdeButton = document.getElementById('drawer-open-ide');

            drawerTitle.textContent = nodeName;
            if (metadata.source_file && metadata.source_start_line) {
                openIdeButton.style.display = 'flex';
                openIdeButton.onclick = function () {
                    const filePath = metadata.source_file;
                    const lineNum = metadata.source_start_line;

                    function detectIDE() {
                        const savedIDE = localStorage.getItem('preferred_ide');
                        if (savedIDE) return savedIDE;

                        if (navigator.userAgent.includes('JetBrains')) return 'jetbrains';

                        return 'auto';
                    }

                    const detectedIDE = detectIDE();
                    let ideUrl;

                    if (detectedIDE === 'pycharm' || detectedIDE === 'auto') {
                        ideUrl = `pycharm://open?file=${filePath}&line=${lineNum}`;
                    } else if (detectedIDE === 'vscode') {
                        ideUrl = `vscode://file/${filePath}:${lineNum}`;
                    } else if (detectedIDE === 'jetbrains') {
                        ideUrl = `jetbrains://open?file=${encodeURIComponent(filePath)}&line=${lineNum}`;
                    } else {
                        ideUrl = `pycharm://open?file=${filePath}&line=${lineNum}`;
                    }
                    const link = document.createElement('a');
                    link.href = ideUrl;
                    link.target = '_blank';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);

                    const fallbackText = `${filePath}:${lineNum}`;
                    navigator.clipboard.writeText(fallbackText).then(function () {
                        console.log('File path copied to clipboard:', fallbackText);
                    }).catch(function (err) {
                        console.error('Failed to copy:', err);
                    });
                };
            } else {
                openIdeButton.style.display = 'none';
            }

            let content = '';

            let metadataContent = '';
            const nodeType = metadata.type || 'unknown';
            const typeBadgeColor = nodeType === 'start' || nodeType === 'router' ? '{{ CREWAI_ORANGE }}' : '{{ DARK_GRAY }}';
            metadataContent += `
        <div class="drawer-section">
            <div class="drawer-section-title">Type</div>
            <span class="drawer-badge" style="background: ${typeBadgeColor}; color: white;">${nodeType}</span>
        </div>
    `;

            if (metadata.condition_type) {
                let conditionColor, conditionBg;
                if (metadata.condition_type === 'AND') {
                    conditionColor = '{{ CREWAI_ORANGE }}';
                    conditionBg = 'rgba(255,90,80,0.12)';
                } else if (metadata.condition_type === 'IF') {
                    conditionColor = '{{ CREWAI_ORANGE }}';
                    conditionBg = 'rgba(255,90,80,0.18)';
                } else {
                    conditionColor = '{{ GRAY }}';
                    conditionBg = 'rgba(102,102,102,0.12)';
                }
                metadataContent += `
            <div class="drawer-section">
                <div class="drawer-section-title">Condition</div>
                <span class="drawer-badge" style="background: ${conditionBg}; color: ${conditionColor};">${metadata.condition_type}</span>
            </div>
        `;
            }

            if (metadata.trigger_methods && metadata.trigger_methods.length > 0) {
                metadataContent += `
            <div class="drawer-section">
                <div class="drawer-section-title">Triggered By</div>
                <ul class="drawer-list">
                    ${metadata.trigger_methods.map(t => `<li><span class="drawer-code-link" data-node-id="${t}">${t}</span></li>`).join('')}
                </ul>
            </div>
        `;
            }

            if (metadata.router_paths && metadata.router_paths.length > 0) {
                metadataContent += `
            <div class="drawer-section">
                <div class="drawer-section-title">Router Paths</div>
                <ul class="drawer-list">
                    ${metadata.router_paths.map(p => `<li><span class="drawer-code-link" data-node-id="${p}" style="color: {{ CREWAI_ORANGE }}; border-color: rgba(255,90,80,0.3);">${p}</span></li>`).join('')}
                </ul>
            </div>
        `;
            }

            if (metadataContent) {
                content += `<div class="drawer-metadata-grid">${metadataContent}</div>`;
            }

            if (metadata.source_code) {
                let lines = metadata.source_lines || metadata.source_code.split('\n');
                if (metadata.source_lines) {
                    lines = lines.map(function(line) { return line.replace(/\n$/, ''); });
                }

                let minIndent = Infinity;
                lines.forEach(function (line) {
                    if (line.trim().length > 0) {
                        const match = line.match(/^\s*/);
                        const indent = match ? match[0].length : 0;
                        minIndent = Math.min(minIndent, indent);
                    }
                });

                const dedentedLines = lines.map(function (line) {
                    if (line.trim().length === 0) return '';
                    return line.substring(minIndent);
                });

                const startLine = metadata.source_start_line || 1;

                const codeToHighlight = dedentedLines.join('\n').trim();

                const highlightedCode = highlightPython(codeToHighlight);

                const highlightedLines = highlightedCode.split('\n');
                const numberedLines = highlightedLines.map(function (line, index) {
                    const lineNum = startLine + index;
                    return '<div class="code-line"><span class="line-number">' + lineNum + '</span>' + line + '</div>';
                }).join('');

                const titleText = 'Source Code';

                let classSection = '';
                if (metadata.class_signature) {
                    const highlightedClass = highlightPython(metadata.class_signature);

                    let highlightedClassSignature = highlightedClass;
                    if (metadata.class_line_number) {
                        highlightedClassSignature = '<span class="line-number">' + metadata.class_line_number + '</span>' + highlightedClass;
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

                content += `
            <div class="accordion-section expanded">
                <div class="accordion-header">
                    <div class="accordion-title">${titleText}</div>
                    <svg class="accordion-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="4,6 8,10 12,6" stroke-linecap="round" stroke-linejoin="round" />
                    </svg>
                </div>
                <div class="accordion-content">
                    <div class="drawer-section">
                        <div class="code-block-container">
                            <button class="code-copy-button" data-code="${codeToHighlight.replace(/"/g, '&quot;')}">
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

            drawerContent.innerHTML = content;

            // Add click handlers for copy buttons
            const copyButtons = drawerContent.querySelectorAll('.code-copy-button');
            copyButtons.forEach(function (button) {
                button.addEventListener('click', function () {
                    const codeText = button.getAttribute('data-code');

                    // Copy to clipboard
                    navigator.clipboard.writeText(codeText).then(function () {
                        // Show success feedback with icon transition
                        button.classList.add('copied');

                        // Reset after 2 seconds
                        setTimeout(function () {
                            button.classList.remove('copied');
                        }, 2000);
                    }).catch(function (err) {
                        console.error('Failed to copy:', err);
                    });
                });
            });

            // Add accordion event handlers
            const accordionHeaders = drawerContent.querySelectorAll('.accordion-header');
            accordionHeaders.forEach(function (header) {
                header.addEventListener('click', function () {
                    const accordionSection = header.closest('.accordion-section');

                    // Toggle the expanded class
                    accordionSection.classList.toggle('expanded');
                });
            });

            // Add click handlers for trigger method links
            const triggerLinks = drawerContent.querySelectorAll('.drawer-code-link');
            triggerLinks.forEach(function (link) {
                link.addEventListener('click', function () {
                    const targetNodeId = link.getAttribute('data-node-id');
                    const currentNodeId = nodeName;

                    if (targetNodeId) {
                        console.log('Clicked on:', targetNodeId, 'from:', currentNodeId);

                        // Clear any existing highlights
                        clearHighlights();

                        // Find the edge connecting these nodes
                        // The edge direction depends on what was clicked:
                        // - If clicking a "Triggered By" item: edge goes FROM that trigger TO current node
                        // - If clicking a "Router Path": edge goes FROM current node (router) TO the path listener

                        let edge = edges.get().find(function (e) {
                            return e.from === targetNodeId && e.to === currentNodeId;
                        });

                        // Store the direction for animation
                        let isForwardAnimation = false;  // Default: animate from trigger to current (backward in graph)

                        // If not found, check if this is a router path (edge FROM currentNode TO target that's triggered by this path)
                        if (!edge) {
                            edge = edges.get().find(function (e) {
                                return e.from === currentNodeId && e.to === targetNodeId;
                            });
                            isForwardAnimation = true;  // Animate from current to target (forward in graph)
                        }

                        console.log('Found edge:', edge, 'Forward animation:', isForwardAnimation);

                        // Check if the targetNodeId is an actual node, or if we need to find the node that listens to this path
                        let actualTargetNodeId = targetNodeId;
                        const targetNode = nodes.get(targetNodeId);

                        if (!targetNode) {
                            // This might be a router path name, find the node that listens to it
                            console.log('Target is not a node, looking for listener of path:', targetNodeId);

                            // Get the node data to find which node has this path in trigger_methods
                            const allNodesData = '{{ nodeData }}';

                            // Find the node that has this path in its trigger_methods
                            for (const nodeId in allNodesData) {
                                const nodeMetadata = allNodesData[nodeId];
                                if (nodeMetadata.trigger_methods && nodeMetadata.trigger_methods.includes(targetNodeId)) {
                                    actualTargetNodeId = nodeId;
                                    console.log('Found node that listens to path:', actualTargetNodeId);

                                    // Now find the edge from current router to this node
                                    const allEdges = edges.get();
                                    edge = allEdges.find(function (e) {
                                        return e.from === currentNodeId && e.to === actualTargetNodeId;
                                    });

                                    if (edge) {
                                        console.log('Found router edge to listener:', edge);
                                    }
                                    break;
                                }
                            }
                        }

                        // Declare arrays for tracking hidden elements (needed for restore later)
                        let nodesToHide = [];
                        let edgesToHide = [];
                        let edgesToRestore = [];

                        // Animate glow "shooting" from current node to target node
                        const nodeData = nodes.get(actualTargetNodeId);
                        if (nodeData) {
                            // Clear any existing highlights first
                            clearHighlights();

                            // Determine animation source and target based on direction FIRST
                            // For forward animation (router -> listener): source = currentNodeId, target = actualTargetNodeId
                            // For backward animation (trigger -> current): source = actualTargetNodeId, target = currentNodeId
                            const animationSourceId = isForwardAnimation ? currentNodeId : actualTargetNodeId;
                            const animationTargetId = isForwardAnimation ? actualTargetNodeId : currentNodeId;

                            console.log('Animation direction - Source:', animationSourceId, 'Target:', animationTargetId);

                            // Dim all non-animated nodes by reducing their opacity
                            // Store original opacity for all nodes so we can restore them
                            const allNodes = nodes.get();
                            const originalNodeOpacities = new Map();

                            allNodes.forEach(node => {
                                // Store original opacity (default is 1 if not set)
                                originalNodeOpacities.set(node.id, node.opacity !== undefined ? node.opacity : 1);

                                // Dim nodes that aren't part of the animation, keep animated ones at full opacity
                                if (node.id !== animationSourceId && node.id !== animationTargetId) {
                                    nodes.update({
                                        id: node.id,
                                        opacity: 0.8
                                    });
                                } else {
                                    // Keep animated nodes at full opacity (ctxRenderer handles highlighting)
                                    nodes.update({
                                        id: node.id,
                                        opacity: 1.0
                                    });
                                }
                            });

                            const sourceNodeData = nodes.get(animationSourceId);
                            const targetNodeData = nodes.get(animationTargetId);
                            const sourceOriginalShadow = sourceNodeData ? sourceNodeData.shadow : null;

                            originalNodeData = {
                                shadow: targetNodeData ? targetNodeData.shadow : null,
                                opacity: targetNodeData ? targetNodeData.opacity : 1,
                                originalOpacities: originalNodeOpacities
                            };
                            originalSourceNodeData = {
                                shadow: sourceOriginalShadow
                            };
                            highlightedNodeId = animationTargetId;
                            highlightedSourceNodeId = animationSourceId;
                            isAnimating = true;

                            // Phase 1: Pulse the source node (150ms)
                            const phase1Duration = 150;
                            const phase1Start = Date.now();

                            function animateSourcePulse() {
                                if (!isAnimating) return; // Stop if animation was cancelled

                                const elapsed = Date.now() - phase1Start;
                                const progress = Math.min(elapsed / phase1Duration, 1);

                                // Ease in-out for pulse
                                const eased = progress < 0.5
                                    ? 2 * progress * progress
                                    : 1 - Math.pow(-2 * progress + 2, 2) / 2;

                                const pulseSize = eased * 15;

                                nodes.update({
                                    id: animationSourceId,
                                    shadow: {
                                        enabled: true,
                                        color: '{{ CREWAI_ORANGE }}',
                                        size: pulseSize,
                                        x: 0,
                                        y: 0
                                    }
                                });

                                if (progress < 1 && isAnimating) {
                                    requestAnimationFrame(animateSourcePulse);
                                } else {
                                    // Reset source node
                                    if (sourceOriginalShadow) {
                                        nodes.update({
                                            id: animationSourceId,
                                            shadow: sourceOriginalShadow
                                        });
                                    } else {
                                        nodes.update({
                                            id: animationSourceId,
                                            shadow: {
                                                enabled: true,
                                                color: 'rgba(0,0,0,0.1)',
                                                size: 8,
                                                x: 2,
                                                y: 2
                                            }
                                        });
                                    }

                                    // Phase 2: Animate edge and target node together
                                    startPhase2();
                                }
                            }

                            function startPhase2() {
                                const phase2Duration = 400;
                                const phase2Start = Date.now();

                                // Store edge info for clearing
                                if (edge) {
                                    originalEdgeData = {
                                        shadow: edge.shadow,
                                        level: edge.level
                                    };
                                    highlightedEdgeId = edge.id;
                                }

                                function animateTravel() {
                                    if (!isAnimating) return; // Stop if animation was cancelled

                                    const elapsed = Date.now() - phase2Start;
                                    const progress = Math.min(elapsed / phase2Duration, 1);

                                    // Ease-out cubic for smooth arrival
                                    const eased = 1 - Math.pow(1 - progress, 3);

                                    // Animate edge glow
                                    if (edge) {
                                        const edgeGlowSize = eased * 15;
                                        edges.update({
                                            id: edge.id,
                                            shadow: {
                                                enabled: true,
                                                color: '{{ CREWAI_ORANGE }}',
                                                size: edgeGlowSize,
                                                x: 0,
                                                y: 0
                                            }
                                        });
                                    }

                                    // Animate target node glow (starts after edge is halfway)
                                    if (progress > 0.3) {
                                        const nodeProgress = (progress - 0.3) / 0.7;
                                        const nodeEased = 1 - Math.pow(1 - nodeProgress, 3);
                                        const glowSize = nodeEased * 25;

                                        nodes.update({
                                            id: animationTargetId,
                                            shadow: {
                                                enabled: true,
                                                color: '{{ CREWAI_ORANGE }}',
                                                size: glowSize,
                                                x: 0,
                                                y: 0
                                            }
                                        });
                                    }

                                    if (progress < 1 && isAnimating) {
                                        requestAnimationFrame(animateTravel);
                                    }
                                }

                                animateTravel();
                            }

                            animateSourcePulse();
                        } else if (edge) {
                            // If we only have an edge (no target node found), still animate it
                            clearHighlights();

                            originalEdgeData = {
                                shadow: edge.shadow
                            };
                            highlightedEdgeId = edge.id;
                            isAnimating = true;

                            const animationDuration = 300;
                            const startTime = Date.now();

                            function animateEdgeGlow() {
                                if (!isAnimating) return;

                                const elapsed = Date.now() - startTime;
                                const progress = Math.min(elapsed / animationDuration, 1);
                                const eased = 1 - Math.pow(1 - progress, 3);
                                const edgeGlowSize = eased * 15;

                                edges.update({
                                    id: edge.id,
                                    shadow: {
                                        enabled: true,
                                        color: '{{ CREWAI_ORANGE }}',
                                        size: edgeGlowSize,
                                        x: 0,
                                        y: 0
                                    }
                                });

                                if (progress < 1 && isAnimating) {
                                    requestAnimationFrame(animateEdgeGlow);
                                }
                            }

                            animateEdgeGlow();
                        }

                        // Highlights remain visible until user clicks elsewhere or closes drawer
                    }
                });
            });

            // Add click handlers for "Triggered By" links
            const triggeredByLinks = drawerContent.querySelectorAll('.drawer-code-link[data-node-id]');
            triggeredByLinks.forEach(function (link) {
                link.addEventListener('click', function (e) {
                    e.preventDefault();
                    e.stopPropagation();
                    const triggerNodeId = this.getAttribute('data-node-id');
                    highlightTriggeredBy(triggerNodeId);
                });
            });

            // Make drawer visible then open it
            drawer.style.visibility = 'visible';

            // Check if drawer was already open
            const wasAlreadyOpen = drawer.classList.contains('open');

            // Use requestAnimationFrame to ensure visibility change is applied before adding 'open'
            requestAnimationFrame(function () {
                drawer.classList.add('open');
                overlay.classList.add('visible');
                // Move navigation buttons left when drawer opens
                document.querySelector('.nav-controls').classList.add('drawer-open');

                // Only adjust viewport if drawer wasn't already open
                if (!wasAlreadyOpen) {
                    // Adjust network viewport to accommodate drawer after animation completes
                    setTimeout(function () {
                        // Get current viewport
                        const currentScale = network.getScale();
                        const currentPosition = network.getViewPosition();

                        // Calculate the offset needed to shift the graph right
                        // Drawer width is 400px, so shift the graph 120px right (30% of drawer width)
                        const drawerWidth = 400;
                        const offsetX = (drawerWidth * 0.3) / currentScale;

                        // Move the graph to the right to center it in the remaining space
                        network.moveTo({
                            position: {
                                x: currentPosition.x + offsetX,
                                y: currentPosition.y
                            },
                            scale: currentScale,
                            animation: {
                                duration: 300,
                                easingFunction: 'easeInOutQuad'
                            }
                        });
                    }, 50); // Small delay to let drawer animation start
                }
            });
        }

        function closeDrawer() {
            const drawer = document.getElementById('drawer');
            const overlay = document.getElementById('drawer-overlay');
            drawer.classList.remove('open');
            overlay.classList.remove('visible');
            // Move navigation buttons back to right when drawer closes
            document.querySelector('.nav-controls').classList.remove('drawer-open');

            // Clear triggered-by highlights
            clearTriggeredByHighlight();

            // Clear active drawer node's edge highlights with smooth animation
            if (activeDrawerNodeId) {
                activeDrawerEdges.forEach(function (edgeId) {
                    animateEdgeWidth(edgeId, 2, 200);
                });
                activeDrawerNodeId = null;
                activeDrawerEdges = [];

                // Trigger redraw to remove active node highlight
                network.redraw();
            }

            // Adjust network viewport to accommodate drawer closing
            setTimeout(function () {
                network.fit({
                    animation: {
                        duration: 300,
                        easingFunction: 'easeInOutQuad'
                    }
                });
            }, 50);

            // Hide drawer after animation completes
            setTimeout(function () {
                if (!drawer.classList.contains('open')) {
                    drawer.style.visibility = 'hidden';
                }
            }, 300);
        }

// Close drawer on overlay click and also clear highlights
        document.getElementById('drawer-overlay').addEventListener('click', function () {
            closeDrawer();
            clearHighlights();
        });
        document.getElementById('drawer-close').addEventListener('click', closeDrawer);

// Close drawer on Escape key
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                closeDrawer();
            }
        });

// Fit to view after stabilization
        network.once('stabilizationIterationsDone', function () {
            network.fit();
        });

// Navigation buttons - no hover effects due to canvas overlay issues

        document.getElementById('zoom-in').addEventListener('click', function () {
            const scale = network.getScale();
            network.moveTo({
                scale: scale * 1.2,
                animation: {
                    duration: 200,
                    easingFunction: 'easeInOutQuad'
                }
            });
        });

        document.getElementById('zoom-out').addEventListener('click', function () {
            const scale = network.getScale();
            network.moveTo({
                scale: scale * 0.8,
                animation: {
                    duration: 200,
                    easingFunction: 'easeInOutQuad'
                }
            });
        });

        document.getElementById('fit').addEventListener('click', function () {
            network.fit({
                animation: {
                    duration: 300,
                    easingFunction: 'easeInOutQuad'
                }
            });
        });

// Export to PNG
        document.getElementById('export-png').addEventListener('click', function () {
            // Use html2canvas to capture the network
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
            script.onload = function () {
                html2canvas(document.getElementById('network-container')).then(function (canvas) {
                    const link = document.createElement('a');
                    link.download = 'flow_dag.png';
                    link.href = canvas.toDataURL();
                    link.click();
                });
            };
            document.head.appendChild(script);
        });

// Export to PDF
        document.getElementById('export-pdf').addEventListener('click', function () {
            // Use jsPDF to create PDF
            const script1 = document.createElement('script');
            script1.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
            script1.onload = function () {
                const script2 = document.createElement('script');
                script2.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
                script2.onload = function () {
                    html2canvas(document.getElementById('network-container')).then(function (canvas) {
                        const imgData = canvas.toDataURL('image/png');
                        const {jsPDF} = window.jspdf;
                        const pdf = new jsPDF({
                            orientation: canvas.width > canvas.height ? 'landscape' : 'portrait',
                            unit: 'px',
                            format: [canvas.width, canvas.height]
                        });
                        pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
                        pdf.save('flow_dag.pdf');
                    });
                };
                document.head.appendChild(script2);
            };
            document.head.appendChild(script1);
        });

// Export to JSON
        document.getElementById('export-json').addEventListener('click', function () {
            const dagData = '{{ dagData }}';
            const dataStr = JSON.stringify(dagData, null, 2);
            const blob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.download = 'flow_dag.json';
            link.href = url;
            link.click();
            URL.revokeObjectURL(url);
        });

// Theme toggle functionality
        const themeToggle = document.getElementById('theme-toggle');
        const htmlElement = document.documentElement;

// Function to get current CSS variable value
        function getCSSVariable(name) {
            return getComputedStyle(htmlElement).getPropertyValue(name).trim();
        }

// Function to update edge label colors based on current theme
        function updateEdgeLabelColors() {
            // Keep labels transparent regardless of theme
            edges.forEach(function (edge) {
                edges.update({
                    id: edge.id,
                    font: {
                        color: 'transparent',
                        background: 'transparent'
                    }
                });
            });
        }

// Load saved theme preference from localStorage or default to light
        const savedTheme = localStorage.getItem('theme') || 'light';
        if (savedTheme === 'dark') {
            htmlElement.setAttribute('data-theme', 'dark');
            themeToggle.textContent = '☀️';
            themeToggle.title = 'Toggle Light Mode';
            // Update edge colors after theme is set
            setTimeout(updateEdgeLabelColors, 0);
        }

// Toggle theme on button click
        themeToggle.addEventListener('click', function () {
            const currentTheme = htmlElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            // Update theme
            if (newTheme === 'dark') {
                htmlElement.setAttribute('data-theme', 'dark');
                themeToggle.textContent = '☀️';
                themeToggle.title = 'Toggle Light Mode';
            } else {
                htmlElement.removeAttribute('data-theme');
                themeToggle.textContent = '🌙';
                themeToggle.title = 'Toggle Dark Mode';
            }

            // Save preference to localStorage
            localStorage.setItem('theme', newTheme);

            // Update edge label colors to match new theme
            // Small delay to let CSS variables update
            setTimeout(updateEdgeLabelColors, 50);
        });
    } catch (e) {
        console.log(e);
    }
})()
