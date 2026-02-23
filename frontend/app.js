/**
 * WebSocket Video Stream Client with Events Viewer
 * Connects to FastAPI server and displays real-time object tracking
 */

class VideoStreamClient {
    constructor() {
        // WebSocket connection
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.isManualDisconnect = false;

        // Canvas and context
        this.canvas = document.getElementById('videoCanvas');
        this.ctx = this.canvas.getContext('2d');

        // DOM elements
        this.elements = {
            connectionStatus: document.getElementById('connectionStatus'),
            videoOverlay: document.getElementById('videoOverlay'),
            fps: document.getElementById('fps'),
            frameCount: document.getElementById('frameCount'),
            latency: document.getElementById('latency'),
            objectCount: document.getElementById('objectCount'),
            configInfo: document.getElementById('configInfo'),
            connectBtn: document.getElementById('connectBtn'),
            disconnectBtn: document.getElementById('disconnectBtn'),
            recordingStatus: document.getElementById('recordingStatus'),
            activeTracks: document.getElementById('activeTracks'),
            activeTrackCount: document.getElementById('activeTrackCount'),
            detailModal: document.getElementById('detailModal'),
            closeModal: document.getElementById('closeModal'),
            modalBody: document.getElementById('modalBody'),
            modalTitle: document.getElementById('modalTitle'),
            historyTracks: document.getElementById('historyTracks'),
            historySpeciesCount: document.getElementById('historySpeciesCount'),
            vlmQweenCloudBtn: document.getElementById('vlmQweenCloudBtn'),
            vlmStatusText: document.getElementById('vlmStatusText')
        };

        // Track state
        this.tracks = new Map();

        // Stats
        this.frameReceived = 0;
        this.lastFrameTime = Date.now();

        // Event listeners
        this.setupEventListeners();

        // Initialize VLM mode
        this.fetchVlmMode();

        // Start history polling (every 30 seconds)
        this.fetchHistory();
        setInterval(() => this.fetchHistory(), 30000);

        setInterval(() => this.fetchHistory(), 30000);
    }

    setupEventListeners() {
        this.elements.connectBtn.addEventListener('click', () => this.connect());
        this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());

        // Modal close handlers
        this.elements.closeModal.addEventListener('click', () => this.hideModal());
        window.addEventListener('click', (e) => {
            if (e.target === this.elements.detailModal) this.hideModal();
        });

        // VLM Mode toggles (Locked to Qween)
        if (this.elements.vlmQweenCloudBtn) {
            this.elements.vlmQweenCloudBtn.addEventListener('click', () => this.updateVlmMode('qween_cloud'));
        }
    }

    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('Already connected');
            return;
        }

        this.isManualDisconnect = false;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        console.log(`Connecting to ${wsUrl}...`);
        this.updateConnectionStatus('connecting', 'Connecting...');

        try {
            this.ws = new WebSocket(wsUrl);
            this.ws.onopen = this.onOpen.bind(this);
            this.ws.onmessage = this.onMessage.bind(this);
            this.ws.onerror = this.onError.bind(this);
            this.ws.onclose = this.onClose.bind(this);
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.updateConnectionStatus('error', 'Connection Failed');
        }
    }

    disconnect() {
        this.isManualDisconnect = true;
        if (this.ws) {
            this.ws.close();
        }
        this.updateConnectionStatus('disconnected', 'Disconnected');
        this.elements.connectBtn.disabled = false;
        this.elements.disconnectBtn.disabled = true;
    }

    onOpen(event) {
        console.log('✓ WebSocket connected');
        this.reconnectAttempts = 0;
        this.updateConnectionStatus('connected', 'Connected');
        this.elements.connectBtn.disabled = true;
        this.elements.disconnectBtn.disabled = false;
        this.elements.videoOverlay.classList.add('hidden');
        this.canvas.classList.add('active');
    }

    onMessage(event) {
        try {
            const data = JSON.parse(event.data);

            switch (data.type) {
                case 'config':
                    this.displayConfig(data.data);
                    break;

                case 'frame':
                    this.displayFrame(data.image, data.metadata);
                    break;

                case 'track_new':
                    this.handleTrackNew(data.data);
                    break;

                case 'track_updated':
                    console.log("the ai part is not visible to the ui - debug: track_updated received", data);
                    this.handleTrackUpdated(data.data);
                    break;

                case 'track_removed':
                    this.handleTrackRemoved(data.data);
                    break;

                case 'vlm_mode_updated':
                    this.setVlmUiState(data.data.mode);
                    break;

                case 'error':
                    console.error('Server error:', data.message);
                    break;

                default:
                    console.warn('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error processing message:', error);
        }
    }

    onError(event) {
        console.error('WebSocket error:', event);
        this.updateConnectionStatus('error', 'Connection Error');
    }

    onClose(event) {
        console.log('WebSocket closed:', event.code, event.reason);
        this.canvas.classList.remove('active');
        this.elements.videoOverlay.classList.remove('hidden');
        this.elements.connectBtn.disabled = false;
        this.elements.disconnectBtn.disabled = true;

        // Auto-reconnect if not manual disconnect
        if (!this.isManualDisconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            this.updateConnectionStatus('reconnecting', `Reconnecting (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => this.connect(), delay);
        } else {
            this.updateConnectionStatus('disconnected', 'Disconnected');
        }
    }

    displayFrame(base64Image, metadata) {
        const img = new Image();

        img.onload = () => {
            // Resize canvas to match image
            if (this.canvas.width !== img.width || this.canvas.height !== img.height) {
                this.canvas.width = img.width;
                this.canvas.height = img.height;
            }

            // Draw image on canvas
            this.ctx.drawImage(img, 0, 0);

            // Update stats
            this.updateStats(metadata);

            // Calculate latency
            const now = Date.now();
            const latency = now - this.lastFrameTime;
            this.lastFrameTime = now;
            this.elements.latency.textContent = `${latency} ms`;

            this.frameReceived++;
        };

        img.onerror = (error) => {
            console.error('Error loading image:', error);
        };

        img.src = `data:image/jpeg;base64,${base64Image}`;
    }

    updateStats(metadata) {
        this.elements.fps.textContent = metadata.fps || 0;
        this.elements.frameCount.textContent = metadata.frame_count || 0;
        this.elements.objectCount.textContent = metadata.num_detections || 0;

        // Update recording status
        if (metadata.is_recording && metadata.recording_info) {
            const info = metadata.recording_info;
            if (info.active_recordings_count > 0) {
                const html = `
                    <p class="status-active">🔴 Recording ${info.active_recordings_count} track(s)</p>
                    <div style="margin-top: 10px; font-size: 0.85rem; color: var(--text-secondary);">
                        Track IDs: ${info.recording_ids.join(', ')}
                    </div>
                `;
                this.elements.recordingStatus.innerHTML = html;
            } else {
                this.elements.recordingStatus.innerHTML = '<p class="status-inactive">No active recordings</p>';
            }
        } else {
            this.elements.recordingStatus.innerHTML = '<p class="status-inactive">No active recordings</p>';
        }

        // Update active tracks count
        this.elements.activeTrackCount.textContent = this.tracks.size;
        if (this.tracks.size === 0) {
            if (!this.elements.activeTracks.querySelector('.status-inactive')) {
                this.elements.activeTracks.innerHTML = '<p class="status-inactive">No objects currently tracked</p>';
            }
        } else {
            const inactiveMsg = this.elements.activeTracks.querySelector('.status-inactive');
            if (inactiveMsg) inactiveMsg.remove();
        }
    }

    handleTrackNew(data) {
        console.log(`🆕 Track new: ID=${data.track_id}, Class=${data.class_name}`);
        this.tracks.set(data.track_id, data);
        this.renderTrackCard(data);
    }

    handleTrackUpdated(data) {
        console.log(`🔄 Track updated: ID=${data.track_id}`);
        const track = this.tracks.get(data.track_id);
        if (track) {
            Object.assign(track, data);

            // NEW: If AI confirms it's not an animal OR identifies it as a person, remove it from UI
            if (track.ai_info && (track.ai_info.is_animal === false || track.ai_info.is_person === true)) {
                console.log(`🚫 Target ${data.track_id} is not an animal or is a person. Removing card.`);
                this.tracks.delete(data.track_id);
                this.removeTrackCard(data.track_id);
                return;
            }

            this.updateTrackCard(data.track_id);
            // Refresh history on update too, just in case
            this.fetchHistory();
        } else {
            console.log("the ai part is not visible to the ui - debug: track not found for update", data.track_id);
        }
    }

    handleTrackRemoved(data) {
        console.log(`👋 Track removed: ID=${data.track_id}`);
        this.tracks.delete(data.track_id);
        this.removeTrackCard(data.track_id);
    }

    renderTrackCard(data) {
        const id = `track-card-${data.track_id}`;
        if (document.getElementById(id)) return;

        const card = document.createElement('div');
        card.id = id;
        card.className = 'tracking-card';
        card.style.cursor = 'pointer';
        card.innerHTML = this.getTrackCardHtml(data);

        // Add click listener for details
        card.addEventListener('click', () => this.showTrackDetails(data.track_id));

        this.elements.activeTracks.prepend(card);
    }

    updateTrackCard(trackId) {
        const card = document.getElementById(`track-card-${trackId}`);
        if (!card) return;

        const data = this.tracks.get(trackId);
        card.innerHTML = this.getTrackCardHtml(data);
    }

    removeTrackCard(trackId) {
        const card = document.getElementById(`track-card-${trackId}`);
        if (card) {
            card.classList.add('removing');
            setTimeout(() => {
                card.remove();
                // Refresh history when a card is removed (likely finished processing)
                this.fetchHistory();
            }, 300);
        }
    }

    async fetchHistory() {
        try {
            const response = await fetch('/api/history');
            if (response.ok) {
                const historyData = await response.json();
                this.renderHistory(historyData);
            }
        } catch (error) {
            console.error('Error fetching history:', error);
        }
    }

    renderHistory(historyData) {
        if (!this.elements.historyTracks) return;

        this.elements.historySpeciesCount.innerText = historyData.length;

        if (historyData.length === 0) {
            this.elements.historyTracks.innerHTML = '<p class="status-inactive">No recent unique species</p>';
            return;
        }

        const html = historyData.map(item => {
            const cardHtml = this.getTrackCardHtml(item);
            return `
                <div class="tracking-card history-card">
                    ${cardHtml}
                </div>
            `;
        }).join('');

        this.elements.historyTracks.innerHTML = html;
    }

    // History Panel Management

    getTrackCardHtml(data) {
        const aiInfo = data.ai_info;
        const hasAi = !!aiInfo;

        let aiHtml = '';
        if (hasAi) {
            if (aiInfo.is_animal === false) {
                aiHtml = `
                    <div class="track-ai-info">
                        <div class="ai-status-row">
                            <span class="status-badge status-unknown">NON-ANIMAL</span>
                        </div>
                        <div style="color: var(--text-secondary); font-style: italic; font-size: 0.85rem; margin-top: 8px;">
                            Identified as a non-animal object.
                        </div>
                    </div>
                `;
            } else {
                // Determine if we should show biological details (hide if N/A or null)
                const showDetails = !!aiInfo.scientificName && aiInfo.scientificName !== 'N/A';

                aiHtml = `
                    <div class="track-ai-info">
                        ${showDetails ? `
                        <div class="ai-status-row">
                            <span class="status-badge status-${(aiInfo.conservationStatus || 'unknown').toLowerCase()}">${aiInfo.conservationStatus || 'Unknown'} Status</span>
                            <span class="${aiInfo.isDangerous ? 'ai-danger-badge' : 'ai-safe-badge'}">
                                ${aiInfo.isDangerous ? '🚩 DANGEROUS' : '🌿 SAFE'}
                            </span>
                        </div>
                        
                        <div class="ai-details-grid">
                            <div class="ai-detail-item">
                                <span class="ai-detail-icon">🧬</span>
                                <div class="ai-detail-text">
                                    <span class="ai-detail-label">Scientific Name</span>
                                    <span class="ai-detail-value italic">${aiInfo.scientificName}</span>
                                </div>
                            </div>
                            ${aiInfo.diet ? `
                            <div class="ai-detail-item">
                                <span class="ai-detail-icon">🍴</span>
                                <div class="ai-detail-text">
                                    <span class="ai-detail-label">Diet</span>
                                    <span class="ai-detail-value">${aiInfo.diet}</span>
                                </div>
                            </div>` : ''}
                            ${aiInfo.lifespan ? `
                            <div class="ai-detail-item">
                                <span class="ai-detail-icon">⏳</span>
                                <div class="ai-detail-text">
                                    <span class="ai-detail-label">Lifespan</span>
                                    <span class="ai-detail-value">${aiInfo.lifespan}</span>
                                </div>
                            </div>` : ''}
                        </div>
                        ` : ''}

                        <div class="ai-description-container">
                           <span class="ai-description-label">Expert Note</span>
                           <p class="ai-description-text">${aiInfo.description}</p>
                        </div>
                    </div>
                `;
            }
        } else {
            aiHtml = `
                <div class="track-ai-info">
                    <div class="ai-loading">
                        <div class="spinner-small"></div>
                        <span>Processing AI info...</span>
                    </div>
                </div>
            `;
        }

        let headerTitle = `🔍 Processing...`;
        if (hasAi) {
            headerTitle = aiInfo.is_animal ? `🔍 ${aiInfo.commonName}` : `🔍 Detected Object`;
        }

        return `
            <div class="track-header">
                <span class="track-class">${headerTitle}</span>
                ${data.track_id ? `<span class="track-id-badge">ID: ${data.track_id}</span>` : ''}
            </div>
            ${data.frame_snapshot ? `
                <div class="track-snapshot">
                    <img src="data:image/jpeg;base64,${data.frame_snapshot}" alt="Track Snapshot">
                </div>
            ` : ''}
            ${aiHtml}
        `;
    }

    showTrackDetails(trackId) {
        const track = this.tracks.get(trackId);
        if (!track || !track.ai_info) return;

        const ai = track.ai_info;
        this.elements.modalTitle.innerText = `🐾 ${ai.commonName} Details`; // Changed to innerText and added emoji, removed non-animal title logic

        let html = '';
        if (ai.is_animal === false) {
            html = `
                <div class="detail-section">
                    ${track.frame_snapshot ? `
                        <div class="detail-image">
                            <img src="data:image/jpeg;base64,${track.frame_snapshot}" alt="Snapshot">
                        </div>
                    ` : ''}
                    <div class="detail-item">
                        <h3>Detection Class</h3>
                        <p>${track.class_name}</p>
                    </div>
                </div>
                <div class="detail-section">
                    <div class="detail-item">
                        <h3>AI Analysis</h3>
                        <p>The system has identified this object as a non-animal. No detailed biological information is available.</p>
                    </div>
                </div>
            `;
        } else {
            // Determine which sections to show (hide if null, empty, or 'N/A')
            const hasBio = !!(ai.scientificName && ai.scientificName !== 'N/A');
            const hasHabitat = !!(ai.habitat && ai.habitat !== 'N/A');
            const hasBehavior = !!(ai.behavior && ai.behavior !== 'N/A');
            const hasSafety = !!(ai.safetyInfo && ai.safetyInfo !== 'N/A' && ai.safetyInfo !== 'null');

            html = `
                <div class="detail-section">
                    ${track.frame_snapshot ? `
                        <div class="detail-image">
                            <img src="data:image/jpeg;base64,${track.frame_snapshot}" alt="Snapshot">
                        </div>
                    ` : ''}
                    ${hasBio ? `
                    <div class="detail-item">
                        <h3>Scientific Name</h3>
                        <p style="font-style: italic;">${ai.scientificName}</p>
                    </div>
                    ` : ''}
                    ${ai.conservationStatus && ai.conservationStatus !== 'Unknown' ? `
                    <div class="detail-item">
                        <h3>Conservation Status</h3>
                        <span class="status-badge status-${ai.conservationStatus}">${ai.conservationStatus}</span>
                    </div>
                    ` : ''}
                    <div class="detail-item">
                        <h3>Safety Status</h3>
                        <p class="${ai.isDangerous ? 'ai-danger' : 'ai-value'}">
                            ${ai.isDangerous ? '⚠️ DANGEROUS - EXERCISE CAUTION' : '✅ Safe / Non-aggressive'}
                        </p>
                    </div>
                </div>
                <div class="detail-section">
                    <div class="detail-item">
                        <h3>Description</h3>
                        <p>${ai.description}</p>
                    </div>
                    <div class="detail-grid">
                        ${hasHabitat ? `
                        <div class="detail-item">
                            <h3>Habitat</h3>
                            <p>${ai.habitat}</p>
                        </div>
                        ` : ''}
                        ${hasBehavior ? `
                        <div class="detail-item">
                            <h3>Behavior</h3>
                            <p>${ai.behavior}</p>
                        </div>
                        ` : ''}
                        ${ai.diet ? `
                        <div class="detail-item">
                            <h3>Diet</h3>
                            <p>${ai.diet}</p>
                        </div>
                        ` : ''}
                        ${ai.lifespan ? `
                        <div class="detail-item">
                            <h3>Lifespan</h3>
                            <p>${ai.lifespan}</p>
                        </div>
                        ` : ''}
                        ${ai.height_cm ? `
                        <div class="detail-item">
                            <h3>Height</h3>
                            <p>${ai.height_cm} cm</p>
                        </div>
                        ` : ''}
                        ${ai.weight_kg ? `
                        <div class="detail-item">
                            <h3>Weight</h3>
                            <p>${ai.weight_kg} kg</p>
                        </div>
                        ` : ''}
                        ${ai.average_speed_kmh ? `
                        <div class="detail-item">
                            <h3>Avg Speed</h3>
                            <p>${ai.average_speed_kmh} km/h</p>
                        </div>
                        ` : ''}
                    </div>
                    ${hasSafety ? `
                    <div class="detail-item" style="border-top: 1px solid var(--border-color); padding-top: 15px;">
                        <h3>Safety Information</h3>
                        <p>${ai.safetyInfo}</p>
                    </div>
                    ` : ''}
                </div>
            `;
        }

        this.elements.modalBody.innerHTML = html;
        this.elements.detailModal.classList.add('active');
    }

    hideModal() {
        this.elements.detailModal.classList.remove('active');
    }

    displayConfig(config) {
        const html = `
            <p><strong>Host:</strong> ${config.host}</p>
            <p><strong>Port:</strong> ${config.port}</p>
            <p><strong>Resolution:</strong> ${config.camera_resolution}</p>
            <p><strong>Target FPS:</strong> ${config.target_fps}</p>
            <p><strong>YOLO Model:</strong> ${config.yolo_model}</p>
            <p><strong>JPEG Quality:</strong> ${config.jpeg_quality}%</p>
        `;
        this.elements.configInfo.innerHTML = html;
    }

    updateConnectionStatus(status, text) {
        const statusDot = this.elements.connectionStatus.querySelector('.status-dot');
        const statusText = this.elements.connectionStatus.querySelector('.status-text');

        statusText.textContent = text;

        // Update dot color
        statusDot.classList.remove('connected');
        if (status === 'connected') {
            statusDot.classList.add('connected');
        }
    }

    // VLM Mode Management
    async fetchVlmMode() {
        try {
            const response = await fetch('/api/config/vlm_mode');
            if (response.ok) {
                const data = await response.json();
                this.setVlmUiState(data.mode);
            }
        } catch (error) {
            console.error('Error fetching VLM mode:', error);
        }
    }

    async updateVlmMode(mode) {
        try {
            const response = await fetch('/api/config/vlm_mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode })
            });
            if (response.ok) {
                this.setVlmUiState(mode);
            }
        } catch (error) {
            console.error('Error updating VLM mode:', error);
        }
    }

    setVlmUiState(mode) {
        console.log(`🧠 AI Status: ${mode}`);
        if (this.elements.vlmQweenCloudBtn) {
            this.elements.vlmQweenCloudBtn.classList.toggle('active', mode === 'qween_cloud');
        }

        const modeNames = {
            'qween_cloud': 'Qween Cloud (VL)'
        };
        this.elements.vlmStatusText.innerHTML = `Status: <strong>${modeNames[mode] || mode} Active</strong>`;
    }
}

// Events Manager
class EventsManager {
    constructor() {
        this.events = [];
        this.filteredEvents = [];
        this.selectedEventId = null;

        this.elements = {
            eventsList: document.getElementById('eventsList'),
            refreshBtn: document.getElementById('refreshEventsBtn'),
            searchInput: document.getElementById('eventSearch'),
            classFilter: document.getElementById('classFilter'),
            videoPlayerArea: document.getElementById('videoPlayerArea'),
            eventMetadata: document.getElementById('eventMetadata')
        };

        this.setupEventListeners();
        this.loadEvents();
    }

    setupEventListeners() {
        this.elements.refreshBtn.addEventListener('click', () => this.loadEvents());
        this.elements.searchInput.addEventListener('input', (e) => this.filterEvents());
        this.elements.classFilter.addEventListener('change', (e) => this.filterEvents());
    }

    async loadEvents() {
        try {
            this.elements.eventsList.innerHTML = '<p class="loading-text">Loading events...</p>';

            const response = await fetch('/api/events');
            if (!response.ok) throw new Error('Failed to load events');

            this.events = await response.json();
            this.updateClassFilter();
            this.filterEvents();

            console.log(`Loaded ${this.events.length} events`);
        } catch (error) {
            console.error('Error loading events:', error);
            this.elements.eventsList.innerHTML = '<p class="loading-text">Error loading events</p>';
        }
    }

    updateClassFilter() {
        const classes = new Set();
        this.events.forEach(event => {
            if (event.detected_classes) {
                event.detected_classes.forEach(cls => classes.add(cls));
            }
        });

        const currentValue = this.elements.classFilter.value;
        this.elements.classFilter.innerHTML = '<option value="">All Classes</option>';

        Array.from(classes).sort().forEach(cls => {
            const option = document.createElement('option');
            option.value = cls;
            option.textContent = cls.charAt(0).toUpperCase() + cls.slice(1);
            this.elements.classFilter.appendChild(option);
        });

        this.elements.classFilter.value = currentValue;
    }

    filterEvents() {
        const searchTerm = this.elements.searchInput.value.toLowerCase();
        const classFilter = this.elements.classFilter.value;

        this.filteredEvents = this.events.filter(event => {
            const matchesSearch = !searchTerm ||
                event.event_id.toLowerCase().includes(searchTerm) ||
                (event.class_name && event.class_name.toLowerCase().includes(searchTerm));

            const matchesClass = !classFilter ||
                (event.detected_classes && event.detected_classes.includes(classFilter));

            return matchesSearch && matchesClass;
        });

        this.renderEvents();
    }

    renderEvents() {
        if (this.filteredEvents.length === 0) {
            this.elements.eventsList.innerHTML = '<p class="loading-text">No events found</p>';
            return;
        }

        const html = this.filteredEvents.map(event => this.createEventCard(event)).join('');
        this.elements.eventsList.innerHTML = html;

        // Add click listeners
        this.elements.eventsList.querySelectorAll('.event-card').forEach(card => {
            card.addEventListener('click', () => {
                const eventId = card.dataset.eventId;
                this.selectEvent(eventId);
            });
        });
    }

    createEventCard(event) {
        const startTime = event.start_time ? new Date(event.start_time).toLocaleString() : 'N/A';
        const duration = event.duration_seconds ? `${event.duration_seconds}s` : 'N/A';

        // Use extracted animals from Florence AI if available, otherwise fall back to YOLO class
        let displayClass = event.class_name || (event.detected_classes && event.detected_classes[0]) || 'Unknown';
        if (event.extracted_animals && event.extracted_animals.length > 0) {
            displayClass = event.extracted_animals.join(', ');
        }

        const path = event.relative_path || event.event_id;
        const isSelected = path === this.selectedEventId ? 'selected' : '';

        return `
            <div class="event-card ${isSelected}" data-event-id="${path}">
                ${event.frame_snapshot ? `
                    <div class="track-snapshot" style="margin-bottom: 10px; height: 120px;">
                        <img src="data:image/jpeg;base64,${event.frame_snapshot}" alt="Event Snapshot">
                    </div>
                ` : ''}
                <div class="event-card-header">
                    <div class="event-title">${event.event_id}</div>
                    <span class="event-class">${displayClass}</span>
                </div>
                <div class="event-details">
                    <div class="event-detail-row">
                        <span>⏱️ Duration:</span>
                        <span>${duration}</span>
                    </div>
                    <div class="event-detail-row">
                        <span>📅 Time:</span>
                        <span>${startTime}</span>
                    </div>
                </div>
            </div>
        `;
    }

    async selectEvent(eventId) {
        this.selectedEventId = eventId;
        this.renderEvents(); // Re-render to update selection

        const event = this.events.find(e => e.event_id === eventId || e.relative_path === eventId);
        if (!event) return;

        const path = event.relative_path || event.event_id;

        try {
            // Load video
            const videoUrl = `/api/events/${path}/video`;
            const videoHtml = `
                <video controls autoplay>
                    <source src="${videoUrl}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            `;
            this.elements.videoPlayerArea.innerHTML = videoHtml;

            // Load and display metadata
            const metadataResponse = await fetch(`/api/events/${path}/metadata`);
            if (metadataResponse.ok) {
                const metadata = await metadataResponse.json();
                this.displayMetadata(metadata);
            }
        } catch (error) {
            console.error('Error loading event:', error);
            this.elements.videoPlayerArea.innerHTML = '<div class="no-selection"><p>Error loading video</p></div>';
        }
    }

    displayMetadata(metadata) {
        const startTime = metadata.start_time ? new Date(metadata.start_time).toLocaleString() : 'N/A';
        const endTime = metadata.end_time ? new Date(metadata.end_time).toLocaleString() : 'N/A';
        const ai = metadata.ai_info;

        let html = `
            <div class="metadata-grid">
                <div class="metadata-item">
                    <span class="metadata-label">Event ID</span>
                    <span class="metadata-value">${metadata.event_id || 'N/A'}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Track ID</span>
                    <span class="metadata-value">${metadata.track_id || 'N/A'}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Class</span>
                    <span class="metadata-value">${metadata.class_name || 'N/A'} ${ai ? `(${ai.commonName})` : ''}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Duration</span>
                    <span class="metadata-value">${metadata.duration_seconds || 'N/A'}s</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Start Time</span>
                    <span class="metadata-value">${startTime}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">End Time</span>
                    <span class="metadata-value">${endTime}</span>
                </div>
            </div>
        `;

        if (ai) {
            html += `
                <div style="margin-top: 20px; border-top: 1px solid var(--border-color); padding-top: 20px;">
                    <h3 style="color: var(--accent-primary); margin-bottom: 15px;">🔍 AI Information</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <h3>Scientific Name</h3>
                            <p style="font-style: italic;">${ai.scientificName}</p>
                        </div>
                        <div class="detail-item">
                            <h3>Status</h3>
                            <span class="status-badge status-${ai.conservationStatus}">${ai.conservationStatus}</span>
                        </div>
                        <div class="detail-item">
                            <h3>Safety Info</h3>
                            <p class="${ai.isDangerous ? 'ai-danger' : 'ai-value'}">${ai.isDangerous ? '⚠️ Dangerous' : '✅ Safe'}</p>
                        </div>
                        <div class="detail-item">
                            <h3>Habitat</h3>
                            <p>${ai.habitat}</p>
                        </div>
                        <div class="detail-item">
                            <h3>Behavior</h3>
                            <p>${ai.behavior}</p>
                        </div>
                        <div class="detail-item">
                            <h3>Description</h3>
                            <p>${ai.description}</p>
                        </div>
                    </div>
                </div>
            `;
        }

        this.elements.eventMetadata.innerHTML = html;
        this.elements.eventMetadata.classList.remove('hidden');
    }
}

// Tab Manager
class TabManager {
    constructor() {
        this.tabs = document.querySelectorAll('.tab-btn');
        this.panes = document.querySelectorAll('.tab-pane');

        this.setupEventListeners();
    }

    setupEventListeners() {
        this.tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const targetTab = tab.dataset.tab;
                this.switchTab(targetTab);
            });
        });
    }

    switchTab(tabName) {
        // Update tab buttons
        this.tabs.forEach(tab => {
            if (tab.dataset.tab === tabName) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });

        // Update tab panes
        this.panes.forEach(pane => {
            if (pane.id === `${tabName}Tab`) {
                pane.classList.add('active');
            } else {
                pane.classList.remove('active');
            }
        });
    }
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Video Stream Client...');

    const streamClient = new VideoStreamClient();
    const eventsManager = new EventsManager();
    const tabManager = new TabManager();
    const chatManager = new ChatManager();

    // Auto-connect on load
    setTimeout(() => streamClient.connect(), 500);
});

// ============================================
// CHATBOT MANAGER
// ============================================

class ChatManager {
    constructor() {
        this.elements = {
            chatMessages: document.getElementById('chatMessages'),
            chatInput: document.getElementById('chatInput'),
            sendBtn: document.getElementById('sendChatBtn'),
            clearBtn: document.getElementById('clearChatBtn')
        };

        this.setupEventListeners();
    }

    setupEventListeners() {
        // Send button click
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());

        // Enter key in input
        this.elements.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Clear chat button
        this.elements.clearBtn.addEventListener('click', () => this.clearChat());

        // Sample query buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('sample-query-btn')) {
                const query = e.target.dataset.query;
                this.elements.chatInput.value = query;
                this.sendMessage();
            }
        });
    }

    async sendMessage() {
        const query = this.elements.chatInput.value.trim();
        if (!query) return;

        // Clear input
        this.elements.chatInput.value = '';

        // Remove welcome message if present
        const welcome = this.elements.chatMessages.querySelector('.chat-welcome');
        if (welcome) welcome.remove();

        // Add user message
        this.addUserMessage(query);

        // Add loading indicator
        const loadingId = this.addLoadingMessage();

        try {
            // Send request to backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });

            const result = await response.json();

            // Remove loading indicator
            this.removeLoadingMessage(loadingId);

            // Add assistant response
            this.addAssistantMessage(result);

        } catch (error) {
            console.error('Chat error:', error);
            this.removeLoadingMessage(loadingId);
            this.addErrorMessage('Failed to process query. Please try again.');
        }

        // Scroll to bottom
        this.scrollToBottom();
    }

    addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message user';
        messageDiv.innerHTML = `
            <div class="chat-message-avatar">👤</div>
            <div class="chat-message-content">
                <div class="chat-message-bubble">
                    <div class="chat-message-text">${this.escapeHtml(text)}</div>
                </div>
                <div class="chat-message-meta">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        this.elements.chatMessages.appendChild(messageDiv);
    }

    addAssistantMessage(result) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message assistant';

        let contentHtml = '';

        // Add main response text
        if (result.messages && result.messages.length > 0) {
            contentHtml += `<div class="chat-message-text">${this.escapeHtml(result.messages[0])}</div>`;
        }

        // Add SQL query if available
        if (result.sql_query) {
            contentHtml += `
                <div class="chat-sql-query">
                    <div class="chat-sql-query-label">Generated SQL:</div>
                    <div class="chat-sql-query-text">${this.escapeHtml(result.sql_query)}</div>
                </div>
            `;
        }

        // Add data table if available
        if (result.data && result.data.length > 0) {
            contentHtml += this.createDataTable(result.data);
            contentHtml += `<div class="chat-row-count">✓ ${result.row_count} row(s) returned</div>`;
        }

        // Add error if present
        if (result.error) {
            contentHtml += `<div class="chat-error">⚠️ ${this.escapeHtml(result.error)}</div>`;
        }

        messageDiv.innerHTML = `
            <div class="chat-message-avatar">🤖</div>
            <div class="chat-message-content">
                <div class="chat-message-bubble">
                    ${contentHtml}
                </div>
                <div class="chat-message-meta">${new Date().toLocaleTimeString()}</div>
            </div>
        `;

        this.elements.chatMessages.appendChild(messageDiv);
    }

    createDataTable(data) {
        if (!data || data.length === 0) return '';

        const columns = Object.keys(data[0]);
        const maxRows = 10; // Limit display to 10 rows
        const displayData = data.slice(0, maxRows);

        let html = '<div class="chat-data-table"><table><thead><tr>';

        // Table headers
        columns.forEach(col => {
            html += `<th>${this.escapeHtml(col)}</th>`;
        });
        html += '</tr></thead><tbody>';

        // Table rows
        displayData.forEach(row => {
            html += '<tr>';
            columns.forEach(col => {
                const value = row[col];
                html += `<td>${value !== null && value !== undefined ? this.escapeHtml(String(value)) : '-'}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table></div>';

        if (data.length > maxRows) {
            html += `<div class="chat-message-meta" style="margin-top: 8px; text-align: center;">Showing ${maxRows} of ${data.length} rows</div>`;
        }

        return html;
    }

    addLoadingMessage() {
        const loadingId = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.id = loadingId;
        loadingDiv.className = 'chat-message assistant';
        loadingDiv.innerHTML = `
            <div class="chat-message-avatar">🤖</div>
            <div class="chat-message-content">
                <div class="chat-loading">
                    <div class="chat-loading-spinner"></div>
                    <span>Processing your query...</span>
                </div>
            </div>
        `;
        this.elements.chatMessages.appendChild(loadingDiv);
        this.scrollToBottom();
        return loadingId;
    }

    removeLoadingMessage(loadingId) {
        const loadingDiv = document.getElementById(loadingId);
        if (loadingDiv) loadingDiv.remove();
    }

    addErrorMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message assistant';
        messageDiv.innerHTML = `
            <div class="chat-message-avatar">🤖</div>
            <div class="chat-message-content">
                <div class="chat-message-bubble">
                    <div class="chat-error">⚠️ ${this.escapeHtml(message)}</div>
                </div>
                <div class="chat-message-meta">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        this.elements.chatMessages.appendChild(messageDiv);
    }

    clearChat() {
        this.elements.chatMessages.innerHTML = `
            <div class="chat-welcome">
                <h3>👋 Welcome to the AI Assistant!</h3>
                <p>Ask me anything about the tracked animals in the database. I can help you query detection data, statistics, and more.</p>
                
                <div class="sample-queries">
                    <p><strong>Try these sample queries:</strong></p>
                    <button class="sample-query-btn" data-query="Show me the last 5 detected animals">Show me the last 5 detected animals</button>
                    <button class="sample-query-btn" data-query="What animals were detected today?">What animals were detected today?</button>
                    <button class="sample-query-btn" data-query="How many unique species have been verified?">How many unique species have been verified?</button>
                    <button class="sample-query-btn" data-query="Show all dangerous animals detected">Show all dangerous animals detected</button>
                </div>
            </div>
        `;
    }

    scrollToBottom() {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
