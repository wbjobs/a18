const socket = io();

let latencyData = {
    labels: [],
    datasets: [{
        label: '平均延迟 (s)',
        data: [],
        borderColor: 'rgb(99, 102, 241)',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointHoverRadius: 6
    }]
};

let latencyChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initLatencyChart();
    loadWatermarkPattern();
});

function initLatencyChart() {
    const ctx = document.getElementById('latencyChart').getContext('2d');
    latencyChart = new Chart(ctx, {
        type: 'line',
        data: latencyData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#e0e0e0'
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#9ca3af',
                        maxRotation: 0,
                        maxTicksLimit: 10
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                },
                y: {
                    ticks: {
                        color: '#9ca3af'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    beginAtZero: true
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

function loadWatermarkPattern() {
    fetch('/api/watermark/pattern')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                drawPattern(data.pattern);
            }
        })
        .catch(error => console.error('Failed to load pattern:', error));
}

function drawPattern(pattern) {
    const canvas = document.getElementById('patternCanvas');
    const ctx = canvas.getContext('2d');
    const size = 5;
    const pixelSize = canvas.width / size;
    
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const r = Math.floor(pattern[i][j][0] * 255);
            const g = Math.floor(pattern[i][j][1] * 255);
            const b = Math.floor(pattern[i][j][2] * 255);
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(j * pixelSize, i * pixelSize, pixelSize, pixelSize);
        }
    }
}

function formatTime(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('zh-CN', { 
        hour: '2-digit', 
        minute: '2-digit',
        second: '2-digit'
    });
}

function updateStats(stats) {
    document.getElementById('totalClients').textContent = stats.total_clients;
    document.getElementById('onlineClients').textContent = `${stats.online_clients} 在线`;
    document.getElementById('offlineClients').textContent = `${stats.offline_clients} 离线`;
    document.getElementById('offlineRate').textContent = `${stats.offline_rate.toFixed(1)}%`;
    document.getElementById('offlineProgress').style.width = `${Math.min(stats.offline_rate, 100)}%`;
    document.getElementById('avgLatency').textContent = `${stats.avg_latency.toFixed(2)}s`;
    document.getElementById('minLatency').textContent = `${stats.min_latency.toFixed(2)}s`;
    document.getElementById('maxLatency').textContent = `${stats.max_latency.toFixed(2)}s`;
    document.getElementById('totalUpdates').textContent = stats.total_updates;
    document.getElementById('totalSamples').textContent = stats.total_samples;
    
    const now = new Date();
    const timeLabel = now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    if (latencyData.labels.length > 20) {
        latencyData.labels.shift();
        latencyData.datasets[0].data.shift();
    }
    
    latencyData.labels.push(timeLabel);
    latencyData.datasets[0].data.push(stats.avg_latency);
    
    if (latencyChart) {
        latencyChart.update('none');
    }
}

function updateClients(clients) {
    const container = document.getElementById('clientsList');
    
    if (clients.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>等待客户端连接...</p></div>';
        return;
    }
    
    let html = '';
    clients.sort((a, b) => {
        if (a.status !== b.status) return a.status === 'online' ? -1 : 1;
        return b.total_updates - a.total_updates;
    });
    
    for (const client of clients) {
        const statusClass = client.status === 'online' ? 'online' : 'offline';
        const device = client.metadata?.device || 'unknown';
        const lastSeen = client.last_update ? formatTime(client.last_update) : '从未';
        
        html += `
            <div class="client-item">
                <div class="client-status ${statusClass}"></div>
                <div class="client-info">
                    <div class="client-id">${client.client_id}</div>
                    <div class="client-meta">
                        ${device} · 回合 ${client.current_round}
                    </div>
                </div>
                <div class="client-stats">
                    <div><span class="updates">${client.total_updates}</span> 次更新</div>
                    <div>${client.total_samples} 样本</div>
                    <div style="font-size: 11px; margin-top: 4px;">
                        延迟: ${client.avg_latency.toFixed(2)}s
                    </div>
                </div>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

function updateContributions(report) {
    const container = document.getElementById('contributionList');
    const reports = Object.values(report.reports || {});
    
    if (reports.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>暂无贡献数据</p></div>';
        return;
    }
    
    reports.sort((a, b) => b.weighted_contribution - a.weighted_contribution);
    
    let html = '';
    for (let i = 0; i < reports.length; i++) {
        const item = reports[i];
        const rankClass = i < 3 ? `rank-${i + 1}` : 'other';
        const contributionPct = (item.weighted_contribution * 100).toFixed(1);
        
        html += `
            <div class="contribution-item">
                <div class="contribution-rank ${rankClass}">${i + 1}</div>
                <div class="contribution-info">
                    <div class="contribution-client">${item.client_id}</div>
                    <div style="font-size: 11px; color: #6b7280;">
                        ${item.total_samples} 样本 · ${item.total_updates} 次更新
                    </div>
                </div>
                <div class="contribution-value">${contributionPct}%</div>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

function addLogEntry(message, type = 'info') {
    const container = document.getElementById('activityLog');
    const now = new Date();
    const timeStr = now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <span class="log-time">[${timeStr}]</span>
        <span class="log-message">${message}</span>
    `;
    
    container.insertBefore(entry, container.firstChild);
    
    while (container.children.length > 50) {
        container.removeChild(container.lastChild);
    }
}

function updateRoundInfo(info) {
    document.getElementById('currentRound').textContent = info.current_round;
}

function verifyWatermark() {
    const resultDiv = document.getElementById('watermarkResult');
    resultDiv.textContent = '验证中...';
    resultDiv.className = 'watermark-result';
    resultDiv.style.display = 'block';
    
    fetch('/api/watermark/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold: 0.8 })
    })
    .then(response => response.json())
    .then(data => {
        const successRate = (data.success_rate * 100).toFixed(1);
        const isStolen = data.is_stolen;
        
        if (isStolen) {
            resultDiv.className = 'watermark-result warning';
            resultDiv.innerHTML = `
                <strong>⚠️ 检测到盗版模型!</strong><br>
                水印触发成功率: ${successRate}%<br>
                该模型可能是未经授权复制
            `;
            addLogEntry(`水印验证完成 - 检测到盗版嫌疑 (成功率: ${successRate}%)', 'warning');
        } else {
            resultDiv.className = 'watermark-result success';
            resultDiv.innerHTML = `
                <strong>✅ 模型验证通过</strong><br>
                水印触发成功率: ${successRate}%<br>
                模型为正版
            `;
            addLogEntry(`水印验证完成 - 模型正版 (成功率: ${successRate}%)', 'success');
        }
    })
    .catch(error => {
        resultDiv.className = 'watermark-result warning';
        resultDiv.textContent = '验证失败: ' + error.message;
    });
}

socket.on('connection_established', function(data) {
    document.getElementById('connectionStatus').textContent = '已连接';
    document.getElementById('connectionStatus').className = 'status-badge online';
    addLogEntry('WebSocket 连接已建立');
});

socket.on('disconnect', function() {
    document.getElementById('connectionStatus').textContent = '连接断开';
    document.getElementById('connectionStatus').className = 'status-badge offline';
    addLogEntry('WebSocket 连接断开', 'warning');
});

socket.on('stats_update', function(stats) {
    updateStats(stats);
});

socket.on('clients_update', function(clients) {
    updateClients(clients);
});

socket.on('contributions_update', function(report) {
    updateContributions(report);
});

socket.on('round_update', function(info) {
    updateRoundInfo(info);
});

socket.on('aggregation_complete', function(data) {
    addLogEntry(`回合 ${data.round} 聚合完成 - ${data.num_clients} 个客户端参与`);
});

socket.on('error', function(data) {
    addLogEntry('错误: ' + data.message, 'error');
});

setInterval(() => {
    if (socket.connected) {
        document.getElementById('connectionStatus').textContent = '已连接';
        document.getElementById('connectionStatus').className = 'status-badge online';
    } else {
        document.getElementById('connectionStatus').textContent = '重连中...';
        document.getElementById('connectionStatus').className = 'status-badge offline';
    }
}, 5000);
