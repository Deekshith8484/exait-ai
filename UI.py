<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EXRT AI Platform Dashboard</title>
    <!-- Load Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Load Chart.js for real graphs -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Load Markdown Parser for nice LLM output -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Use Inter font family -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #10B981; /* Tailwind green-500 */
            --secondary-color: #6366F1; /* Tailwind violet-600 */
            --dark-header: #2C3E50; /* Using dark sidebar color for the header */
            --dark-background: #F4F5F7;
            --ready-color: #10B981;
            --caution-color: #FBBF24;
            --recovery-color: #EF4444;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--dark-background);
        }
        
        .page-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }

        .main-content {
            overflow-y: auto;
            flex-grow: 1; 
        }
        .main-content::-webkit-scrollbar {
            width: 8px;
        }

        .main-content::-webkit-scrollbar-thumb {
            background-color: #D1D5DB;
            border-radius: 4px;
        }

        .main-content::-webkit-scrollbar-track {
            background: #F3F4F6;
        }

        .exrt-logo-icon {
            width: 32px;
            height: 32px;
            /* Add a subtle hover effect for the logo */
            transition: transform 0.2s ease-in-out;
        }
        .exrt-logo-icon:hover {
            transform: scale(1.1);
        }

        /* Modal Transitions */
        .modal {
            transition: opacity 0.25s ease;
        }
        body.modal-active {
            overflow-x: hidden;
            overflow-y: visible !important;
        }

        /* AI Thinking Animation */
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        .loading-shimmer {
            animation: shimmer 2s infinite linear;
            background: linear-gradient(to right, #eff6ff 4%, #dbeafe 25%, #eff6ff 36%);
            background-size: 1000px 100%;
        }

        /* Chat Widget Animations */
        .chat-slide-enter {
            transform: translateY(20px);
            opacity: 0;
        }
        .chat-slide-enter-active {
            transform: translateY(0);
            opacity: 1;
            transition: all 0.3s ease-out;
        }
    </style>
</head>
<body class="page-container text-gray-800">

    <!-- Top Navigation Bar (Header) -->
    <nav id="top-header" class="bg-gray-800 text-white shadow-xl px-6 py-3 flex justify-between items-center z-10 relative" style="background-color: var(--dark-header);">
        
        <!-- Logo and Brand -->
        <div class="flex items-center space-x-3">
            <div class="exrt-logo-icon">
                <!-- Replaced SVG with the provided logo.png -->
                <img src="logo.png" alt="EXRT AI Logo" class="w-full h-full object-contain">
            </div>
            <span class="text-xl font-bold">EXRT AI</span>
        </div>

        <!-- Desktop Navigation Links (Center) -->
        <div class="hidden lg:flex space-x-6 text-sm font-bold" id="main-nav">
            <a href="#overview" class="p-2 rounded-lg transition-colors duration-150 text-green-400 bg-gray-700 hover:bg-gray-700">Home</a>
            <a href="#live-simulator" class="p-2 rounded-lg transition-colors duration-150 hover:bg-gray-700">Live</a>
            <a href="#team-dashboard" class="p-2 rounded-lg transition-colors duration-150 hover:bg-gray-700">Team</a>
            <a href="#personal-health" class="p-2 rounded-lg transition-colors duration-150 text-blue-300 hover:bg-gray-700">Personal Health</a> <!-- NEW LINK -->
            <a href="#business-model" class="p-2 rounded-lg transition-colors duration-150 text-yellow-300 hover:bg-gray-700">Business</a>
        </div>

        <!-- Right Side: Profile & Mobile Menu Toggle -->
        <div class="flex items-center space-x-4">
            
            <!-- Mobile Menu Button -->
            <button id="mobile-menu-btn" class="lg:hidden p-2 rounded-lg hover:bg-gray-700 focus:outline-none">
                <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path></svg>
            </button>

            <!-- Profile Icon -->
            <div class="relative">
                <button id="profile-btn" class="flex items-center p-2 rounded-full bg-gray-700 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-green-400 transition duration-150">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-gray-300" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd"/></svg>
                </button>
                
                <!-- Dropdown Menu -->
                <div id="profile-dropdown" class="absolute right-0 mt-2 w-48 bg-gray-700 rounded-xl shadow-2xl py-2 hidden origin-top-right ring-1 ring-black ring-opacity-5 transition ease-out duration-200 transform scale-95 opacity-0 z-50">
                    <h4 class="px-4 pt-1 pb-2 text-xs uppercase font-semibold text-gray-400">Settings</h4>
                    <a href="javascript:void(0)" class="flex items-center px-4 py-2 text-sm text-gray-200 hover:bg-gray-600 transition-colors duration-150">
                        Preferences
                    </a>
                    <a href="javascript:void(0)" class="flex items-center px-4 py-2 text-sm text-gray-200 hover:bg-gray-600 transition-colors duration-150">
                        Account
                    </a>
                    <div class="border-t border-gray-600 my-1"></div>
                    <button class="flex items-center w-full px-4 py-2 text-sm text-red-400 hover:bg-red-900 transition-colors duration-150">
                        Logout
                    </button>
                </div>
            </div>
        </div>
    </nav>

    <!-- Mobile Navigation Menu (Hidden by default) -->
    <div id="mobile-menu" class="hidden lg:hidden bg-gray-800 text-white p-4 absolute top-16 left-0 w-full z-20 shadow-xl border-t border-gray-700">
        <div class="flex flex-col space-y-2">
            <a href="#overview" class="block p-3 rounded-lg hover:bg-gray-700">Home</a>
            <a href="#live-simulator" class="block p-3 rounded-lg hover:bg-gray-700">Live</a>
            <a href="#team-dashboard" class="block p-3 rounded-lg hover:bg-gray-700">Team</a>
            <a href="#personal-health" class="block p-3 rounded-lg hover:bg-gray-700 text-blue-300">Personal Health</a>
            <a href="#business-model" class="block p-3 rounded-lg hover:bg-gray-700 text-yellow-300">Business</a>
        </div>
    </div>

    <!-- Main Content Area -->
    <main class="p-8 main-content relative">
        
        <header class="mb-8 border-b pb-4">
            <h1 class="text-3xl font-extrabold text-gray-900">Platform Overview</h1>
            <p class="text-gray-500">EXRT AI: AI-Driven Health Monitoring and Performance Optimization.</p>
        </header>

        <!-- Section 1: Overview & Features -->
        <section id="overview" class="mb-12">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <!-- Cards (Static) -->
                <div class="bg-white p-6 rounded-xl shadow-lg border border-gray-200 transition duration-300 hover:shadow-xl hover:-translate-y-1">
                    <div class="text-center">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 mx-auto mb-3 text-green-500" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414L9 3.586V4a1 1 0 012 0v-.414l2.707 2.707a1 1 0 01-1.414 1.414L11 6.414V14a1 1 0 01-2 0V6.414L6.707 7.707a1 1 0 01-1.414-1.414z" clip-rule="evenodd"/></svg>
                        <h2 class="text-xl font-semibold mb-2">Upload & Analysis</h2>
                        <p class="text-sm text-gray-500">Securely upload physiological data for in-depth analysis and reporting.</p>
                    </div>
                </div>
                <div class="bg-white p-6 rounded-xl shadow-lg border border-gray-200 transition duration-300 hover:shadow-xl hover:-translate-y-1">
                    <div class="text-center">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 mx-auto mb-3 text-blue-500" viewBox="0 0 20 20" fill="currentColor"><path d="M7 3a1 1 0 00-1 1v1a1 1 0 002 0V4a1 1 0 00-1-1zM9 4V3a1 1 0 00-2 0v1a1 1 0 002 0zm-5 4H4a1 1 0 000 2h1a1 1 0 000-2zm12 0h-1a1 1 0 000 2h1a1 1 0 000-2zM9 16h2v-1H9v1zM4 14V8a1 1 0 011-1h10a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1z" /></svg>
                        <h2 class="text-xl font-semibold mb-2">Live Simulator</h2>
                        <p class="text-sm text-gray-500">Real-time monitoring and simulation of biometric signals (e.g., ECG).</p>
                    </div>
                </div>
                <div class="bg-white p-6 rounded-xl shadow-lg border border-gray-200 transition duration-300 hover:shadow-xl hover:-translate-y-1">
                    <div class="text-center">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 mx-auto mb-3 text-purple-500" viewBox="0 0 20 20" fill="currentColor"><path d="M10 2a8 8 0 100 16 8 8 0 000-16zM6.5 9.5a1 1 0 11-2 0 1 1 0 012 0zm7.5-1a1 1 0 110 2 1 1 0 010-2zM9 14.5a1 1 0 110 2 1 1 0 010-2zM12.5 9.5a1 1 0 11-2 0 1 1 0 012 0z"/></svg>
                        <h2 class="text-xl font-semibold mb-2">AI-Powered Insights</h2>
                        <p class="text-sm text-gray-500">Predictive readiness levels and tailored recovery/optimization plans.</p>
                    </div>
                </div>
            </div>

            <!-- Readiness Levels -->
            <div class="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
                <h2 class="text-xl font-bold mb-4">Readiness Levels</h2>
                <div class="space-y-3">
                    <div class="flex items-center p-3 rounded-lg text-white font-semibold" style="background-color: var(--ready-color);">
                        <span class="w-24">Ready (85%)</span>
                        <div class="flex-1 h-2 bg-white/30 rounded-full ml-4"><div class="h-2 rounded-full shadow" style="width: 85%; background-color: white;"></div></div>
                    </div>
                    <div class="flex items-center p-3 rounded-lg text-gray-800 font-semibold" style="background-color: var(--caution-color);">
                        <span class="w-24">Caution (50%)</span>
                        <div class="flex-1 h-2 bg-gray-600/30 rounded-full ml-4"><div class="h-2 rounded-full shadow" style="width: 50%; background-color: var(--caution-color);"></div></div>
                        <span class="text-sm ml-4">Monitor fatigue markers.</span>
                    </div>
                    <div class="flex items-center p-3 rounded-lg text-white font-semibold" style="background-color: var(--recovery-color);">
                        <span class="w-24">Recovery (10%)</span>
                        <div class="flex-1 h-2 bg-white/30 rounded-full ml-4"><div class="h-2 rounded-full shadow" style="width: 10%; background-color: white;"></div></div>
                        <span class="text-sm ml-4">Immediate rest required.</span>
                    </div>
                </div>
            </div>
        </section>

        <!-- Section 2: Live Monitoring Dashboard (Dynamic & Animated) -->
        <section id="live-simulator" class="mb-12">
            <div class="flex justify-between items-center mb-4 flex-wrap gap-4">
                <h2 class="text-2xl font-bold flex items-center">
                    Live ECG Monitoring Simulator
                    <span class="ml-3 px-2 py-1 text-xs bg-red-500 text-white rounded-full animate-pulse">LIVE</span>
                </h2>
                <!-- Buttons for Live Section -->
                <div class="flex space-x-3">
                    <button onclick="analyzeLiveVitals()" class="px-4 py-2 bg-gradient-to-r from-red-500 to-pink-600 text-white text-sm font-semibold rounded-lg hover:from-red-600 hover:to-pink-700 transition duration-150 shadow-md flex items-center">
                        <span class="mr-2">⚡</span> Analyze Live Vitals
                    </button>
                    <!-- New Prediction Feature -->
                    <button onclick="predictFatigueTrend()" class="px-4 py-2 bg-gradient-to-r from-purple-500 to-indigo-600 text-white text-sm font-semibold rounded-lg hover:from-purple-600 hover:to-indigo-700 transition duration-150 shadow-md flex items-center">
                        <span class="mr-2">🔮</span> Predict Fatigue Trend
                    </button>
                </div>
            </div>

            <div class="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
                
                <!-- Status Bar -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <div class="flex flex-col p-4 rounded-lg text-white font-bold" style="background-color: var(--primary-color);">
                        <div class="text-lg">Device Connection Status</div>
                        <div class="text-3xl mt-1">Connected</div>
                    </div>
                    <div class="bg-gray-100 p-4 rounded-lg">
                        <div class="text-sm text-gray-500">Signal Quality</div>
                        <div class="text-2xl font-bold text-gray-900">98%</div>
                        <div class="h-1 bg-gray-300 rounded-full mt-2"><div class="h-1 rounded-full" style="width: 98%; background-color: var(--primary-color);"></div></div>
                    </div>
                    <div class="bg-gray-100 p-4 rounded-lg">
                        <div class="text-sm text-gray-500">Battery</div>
                        <div class="text-2xl font-bold text-gray-900">87%</div>
                        <div class="h-1 bg-gray-300 rounded-full mt-2"><div class="h-1 rounded-full" style="width: 87%; background-color: var(--primary-color);"></div></div>
                    </div>
                </div>

                <!-- Dashboard Metrics (Dynamic) -->
                <div class="mb-6">
                    <h3 class="text-xl font-semibold mb-3">Live Monitoring Dashboard</h3>
                    <div class="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
                        <div class="p-3 bg-gray-50 rounded-lg border">
                            <div class="text-xs text-gray-500">Heart Rate (BPM)</div>
                            <div class="text-2xl font-bold" id="live-hr">75</div>
                        </div>
                        <div class="p-3 bg-gray-50 rounded-lg border">
                            <div class="text-xs text-gray-500">Fatigue Score</div>
                            <div class="text-2xl font-bold" id="live-fatigue">9.7%</div>
                        </div>
                        <div class="p-3 bg-gray-50 rounded-lg border">
                            <div class="text-xs text-gray-500">Signal Integrity</div>
                            <div class="text-2xl font-bold">100.0%</div>
                        </div>
                        <div class="p-3 bg-gray-50 rounded-lg border bg-yellow-100">
                            <div class="text-xs text-yellow-700">Readiness Status</div>
                            <div class="text-2xl font-bold text-yellow-700">Fatigued</div>
                        </div>
                        <div class="p-3 bg-gray-50 rounded-lg border">
                            <div class="text-xs text-gray-500">Recovery Needed</div>
                            <div class="text-2xl font-bold">0%</div>
                        </div>
                    </div>
                </div>

                <!-- AI Live Analysis Container (Shared for both Analysis and Prediction) -->
                <div id="live-analysis-container" class="mb-6 hidden">
                    <div class="bg-gradient-to-r from-gray-50 to-indigo-50 p-4 rounded-lg border border-gray-200 shadow-sm relative overflow-hidden">
                        <!-- Header will be dynamically set -->
                        <h4 id="analysis-title" class="text-sm font-bold text-gray-800 mb-2 flex items-center"></h4>
                        <div id="live-analysis-text" class="prose text-sm text-gray-800"></div>
                        
                        <!-- Decorative background element -->
                        <div class="absolute top-0 right-0 -mt-2 -mr-2 w-16 h-16 bg-indigo-500 opacity-10 rounded-full blur-xl"></div>
                    </div>
                </div>

                <!-- Real Chart: Meditation/Stress -->
                <div class="mb-6">
                    <h3 class="text-lg font-semibold mb-2">Meditation Time/Stress Level</h3>
                    <div class="h-48 bg-gray-50 rounded-lg border p-2">
                         <canvas id="stressChart" class="w-full h-full"></canvas>
                    </div>
                </div>
                
                <!-- Real Chart: ECG Signal -->
                <div class="mb-6">
                    <h3 class="text-lg font-semibold mb-2">Realtime ECG Signal</h3>
                    <div class="h-48 bg-gray-50 rounded-lg border p-2">
                         <canvas id="ecgChart" class="w-full h-full"></canvas>
                    </div>
                </div>
            </div>
        </section>

        <!-- Section 3: Team Dashboard (Interactive) -->
        <section id="team-dashboard" class="mb-12">
            <div class="bg-green-600 text-white p-4 rounded-t-xl font-bold text-lg text-center">
                Professional Team Monitoring
            </div>
            <div class="bg-white p-6 rounded-b-xl shadow-lg border border-gray-200">
                <h2 class="text-xl font-bold mb-4">Squad Readiness Summary</h2>
                
                <!-- Summary Metrics -->
                <div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6 text-center">
                    <div class="p-3 bg-gray-50 rounded-lg border">
                        <div class="text-sm text-gray-500">Total Players</div>
                        <div class="text-3xl font-bold">25</div>
                    </div>
                    <div class="p-3 bg-green-100 rounded-lg border border-green-300">
                        <div class="text-sm text-green-700">Ready Players</div>
                        <div class="text-3xl font-bold text-green-700">21</div>
                    </div>
                    <div class="p-3 bg-yellow-100 rounded-lg border border-yellow-300">
                        <div class="text-sm text-yellow-700">Caution Players</div>
                        <div class="text-3xl font-bold text-yellow-700">3</div>
                    </div>
                    <div class="p-3 bg-red-100 rounded-lg border border-red-300">
                        <div class="text-sm text-red-700">Recovery Players</div>
                        <div class="text-3xl font-bold text-red-700">1</div>
                    </div>
                    <div class="p-3 bg-gray-50 rounded-lg border">
                        <div class="text-sm text-gray-500">Avg. Readiness Score</div>
                        <div class="text-3xl font-bold">89.4%</div>
                    </div>
                </div>

                <!-- AI Report Section -->
                <div id="ai-report-container" class="mb-6 hidden">
                    <div class="bg-gradient-to-r from-indigo-50 to-purple-50 p-6 rounded-lg border border-indigo-200">
                        <h4 class="text-lg font-bold text-indigo-900 mb-2 flex items-center">
                            <span class="mr-2">✨</span> AI Squad Analysis
                        </h4>
                        <div id="ai-report-text" class="prose text-sm text-gray-800"></div>
                    </div>
                </div>

                <!-- Player Status Grid (Interactive) -->
                <h3 class="text-lg font-semibold mb-2">Player Status Grid (Click Rows for Details)</h3>
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200" id="players-table">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Player</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Position</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Fatigue</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200">
                            <tr class="hover:bg-gray-50 cursor-pointer transition" onclick="openPlayerModal('Player A', 'Forward', '95%', 'Ready', '3.1%')">
                                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Player A</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Forward</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">95%</td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">Ready</span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">3.1%</td>
                            </tr>
                            <tr class="hover:bg-gray-50 cursor-pointer transition" onclick="openPlayerModal('Player B', 'Midfielder', '45%', 'Caution', '15.5%')">
                                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Player B</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Midfielder</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">45%</td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-yellow-100 text-yellow-800">Caution</span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">15.5%</td>
                            </tr>
                            <tr class="hover:bg-gray-50 cursor-pointer transition" onclick="openPlayerModal('Player C', 'Defender', '10%', 'Recovery', '35.2%')">
                                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Player C</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Defender</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">10%</td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">Recovery</span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">35.2%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="flex space-x-3 mt-6">
                    <button class="px-6 py-2 bg-gray-800 text-white font-semibold rounded-lg hover:bg-gray-700 transition duration-150 shadow-md">
                        Generate Full Team Report
                    </button>
                    <!-- AI Feature Button -->
                    <button onclick="generateTeamAnalysis()" class="px-6 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-semibold rounded-lg hover:from-purple-700 hover:to-indigo-700 transition duration-150 shadow-md flex items-center">
                        <span class="mr-2">✨</span> AI Squad Analyst
                    </button>
                    <!-- NEW Match Strategy Button -->
                    <button onclick="generateMatchStrategy()" class="px-6 py-2 bg-gradient-to-r from-blue-600 to-teal-500 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-teal-600 transition duration-150 shadow-md flex items-center">
                        <span class="mr-2">📋</span> Match Strategy
                    </button>
                </div>
            </div>
        </section>

        <!-- NEW Section: Personal Health (Consumer Focused) -->
        <section id="personal-health" class="mb-12">
            <div class="bg-blue-600 text-white p-4 rounded-t-xl font-bold text-lg text-center flex justify-between items-center px-6">
                <span>Personal Health Dashboard</span>
                <span class="text-xs bg-blue-500 px-2 py-1 rounded">Consumer Edition</span>
            </div>
            <div class="bg-white p-6 rounded-b-xl shadow-lg border border-gray-200">
                
                <!-- Patch Status -->
                <div class="flex items-center justify-between bg-blue-50 p-4 rounded-lg border border-blue-100 mb-6">
                    <div class="flex items-center">
                        <div class="h-3 w-3 bg-green-500 rounded-full animate-pulse mr-3"></div>
                        <div>
                            <h3 class="font-bold text-gray-800">ECG Patch Connected</h3>
                            <p class="text-xs text-gray-500">Battery: 92% | Last Sync: Just now</p>
                        </div>
                    </div>
                    <button class="text-sm text-blue-600 font-semibold hover:underline">Device Settings</button>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    
                    <!-- Metrics Column -->
                    <div class="space-y-4">
                        <div class="p-4 bg-gray-50 rounded-lg border">
                            <h4 class="text-xs text-gray-500 uppercase tracking-wide">Daily HRV</h4>
                            <div class="flex items-baseline mt-1">
                                <span class="text-3xl font-bold text-gray-800">45</span>
                                <span class="text-sm text-gray-500 ml-1">ms</span>
                            </div>
                            <p class="text-xs text-green-600 mt-1">↑ 5% vs yesterday</p>
                        </div>
                        <div class="p-4 bg-gray-50 rounded-lg border">
                            <h4 class="text-xs text-gray-500 uppercase tracking-wide">Stress Balance</h4>
                            <div class="flex items-baseline mt-1">
                                <span class="text-3xl font-bold text-yellow-600">Medium</span>
                            </div>
                            <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                                <div class="bg-yellow-500 h-2 rounded-full" style="width: 60%"></div>
                            </div>
                        </div>
                        <div class="p-4 bg-gray-50 rounded-lg border">
                            <h4 class="text-xs text-gray-500 uppercase tracking-wide">Resting Heart Rate</h4>
                            <div class="flex items-baseline mt-1">
                                <span class="text-3xl font-bold text-gray-800">62</span>
                                <span class="text-sm text-gray-500 ml-1">bpm</span>
                            </div>
                        </div>
                    </div>

                    <!-- Chart Column -->
                    <div class="md:col-span-2">
                        <h4 class="font-bold text-gray-700 mb-2">Weekly Wellness Trend</h4>
                        <div class="h-64 bg-white border rounded-lg p-2">
                            <canvas id="personalTrendChart"></canvas>
                        </div>
                        
                        <!-- AI Wellness Coach -->
                        <div class="mt-4">
                            <button onclick="getWellnessAdvice()" class="w-full py-3 bg-gradient-to-r from-blue-500 to-indigo-500 text-white font-bold rounded-lg hover:from-blue-600 hover:to-indigo-600 transition shadow-md flex justify-center items-center">
                                <span class="mr-2">✨</span> Get Daily Wellness Plan
                            </button>
                            <div id="wellness-response" class="hidden mt-4 p-4 bg-blue-50 border border-blue-100 rounded-lg text-sm text-gray-700 prose prose-blue max-w-none"></div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Section 4: Business Model (Functional ROI) -->
        <section id="business-model" class="mb-12">
            <div class="p-6 rounded-xl shadow-2xl border-4 border-indigo-200 bg-white/95 backdrop-blur-sm">
                
                <h2 class="text-3xl font-extrabold text-indigo-700 mb-6 flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 mr-3 text-yellow-500" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd"/></svg>
                    Investment Thesis & Market Opportunity
                </h2>

                <!-- Projected Impact -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 text-center border-b pb-6 border-indigo-100">
                    <div class="p-4 rounded-xl bg-indigo-600 text-white shadow-lg">
                        <div class="text-5xl font-extrabold">35%</div>
                        <p class="text-sm font-semibold mt-1">Injury Risk Reduction</p>
                    </div>
                    <div class="p-4 rounded-xl bg-indigo-600 text-white shadow-lg">
                        <div class="text-5xl font-extrabold">30%</div>
                        <p class="text-sm font-semibold mt-1">Performance Uplift</p>
                    </div>
                    <div class="p-4 rounded-xl bg-yellow-400 text-gray-900 shadow-lg">
                        <div class="text-5xl font-extrabold">$46M+</div>
                        <p class="text-sm font-semibold mt-1">Annual Client Value</p>
                    </div>
                </div>

                <!-- ROI Calculator -->
                <h3 class="text-2xl font-bold mb-4 text-gray-800">Investor ROI Modeling (Interactive)</h3>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- Input Column -->
                    <div class="md:col-span-1 space-y-4 p-4 bg-gray-50 rounded-lg border shadow-sm">
                        <h4 class="font-bold text-gray-700">Client Cost Input</h4>
                        <div>
                            <label class="block text-sm font-medium text-gray-700">Team Size (e.g., Players)</label>
                            <input type="number" id="teamSizeInput" value="25" class="mt-1 block w-full rounded-md border-indigo-300 shadow-sm p-2 border focus:ring-indigo-500 focus:border-indigo-500 font-mono">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700">Avg. Player Salary (K)</label>
                            <input type="number" id="salaryInput" value="500" class="mt-1 block w-full rounded-md border-indigo-300 shadow-sm p-2 border focus:ring-indigo-500 focus:border-indigo-500 font-mono">
                        </div>
                        <div class="text-xs text-gray-500 mt-2 italic">Try changing these values to see real-time projections!</div>
                    </div>

                    <!-- Proposed Savings/Output Column -->
                    <div class="md:col-span-2 space-y-4">
                        <h4 class="font-bold text-gray-700">Financial Projection: Value Delivered</h4>
                        <div class="grid grid-cols-3 gap-4 text-center">
                            <div class="p-4 bg-green-100 rounded-lg border border-green-300 transition-all duration-300">
                                <div class="text-sm text-green-700">Injury Cost Avoided</div>
                                <div class="text-xl md:text-2xl font-extrabold text-green-800" id="injurySavings">$18,750,000</div>
                            </div>
                            <div class="p-4 bg-green-100 rounded-lg border border-green-300 transition-all duration-300">
                                <div class="text-sm text-green-700">Performance Uplift Value</div>
                                <div class="text-xl md:text-2xl font-extrabold text-green-800" id="performanceValue">$27,450,000</div>
                            </div>
                            <div class="p-4 bg-indigo-100 rounded-lg border border-indigo-300 transition-all duration-300">
                                <div class="text-sm text-indigo-700">ROI Multiple (Annual)</div>
                                <div class="text-xl md:text-2xl font-extrabold text-indigo-800" id="roiMultiple">5.2x</div>
                            </div>
                        </div>
                        <div class="mt-4 p-4 bg-yellow-50 rounded-lg border border-yellow-300 text-sm font-medium text-yellow-800">
                            **Investment Insight:** Our technology generates a <span id="roiText">5.2x</span> return on investment for an average professional sports client annually.
                        </div>
                    </div>
                </div>

                <!-- Call to Action & Memo Generator -->
                 <div class="mt-8 pt-6 border-t border-indigo-100 text-center">
                    <button onclick="generateInvestmentMemo()" class="px-8 py-3 bg-gray-600 text-white font-medium text-lg rounded-full hover:bg-gray-700 transition duration-300 flex items-center justify-center mx-auto">
                        <span class="mr-2">✨</span> Generate AI Investment Memo
                    </button>
                    <!-- Container for Memo -->
                    <div id="investment-memo-container" class="hidden mt-6 text-left max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-lg border border-gray-200">
                        <h4 class="text-2xl font-bold text-indigo-800 mb-4 border-b pb-2">EXRT AI: Strategic Investment Brief</h4>
                        <div id="investment-memo-text" class="prose prose-indigo max-w-none text-gray-700"></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Footer -->
        <footer class="mt-12 py-4 text-center text-xs text-gray-500 border-t border-gray-200">
            &copy; 2025 EXRT AI. Empowering Next-Level Health Monitoring.
        </footer>

    </main>

    <!-- Floating Chat Widget -->
    <div id="chat-widget" class="fixed bottom-6 right-6 z-50 flex flex-col items-end">
        <!-- Chat Window -->
        <div id="chat-window" class="hidden bg-white w-80 md:w-96 rounded-2xl shadow-2xl border border-gray-200 mb-4 overflow-hidden flex flex-col transition-all duration-300 transform origin-bottom-right">
            <div class="bg-indigo-700 text-white p-4 flex justify-between items-center">
                <div class="flex items-center">
                    <div class="bg-white/20 p-1.5 rounded-full mr-2">
                        <span class="text-lg">✨</span>
                    </div>
                    <span class="font-bold">Ask EXRT AI</span>
                </div>
                <button onclick="toggleChat()" class="text-white/80 hover:text-white">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                </button>
            </div>
            <div id="chat-messages" class="h-80 overflow-y-auto p-4 bg-gray-50 space-y-3 text-sm">
                <div class="flex justify-start">
                    <div class="bg-indigo-100 text-indigo-900 p-3 rounded-lg rounded-tl-none max-w-[85%]">
                        Hello! I have access to your full dashboard. Ask me about player fatigue, ROI, or live vitals.
                    </div>
                </div>
            </div>
            <div class="p-3 bg-white border-t flex">
                <input type="text" id="chat-input" placeholder="Type a question..." class="flex-grow px-3 py-2 border rounded-l-lg focus:outline-none focus:border-indigo-500 text-sm" onkeypress="if(event.key === 'Enter') sendChatMessage()">
                <button onclick="sendChatMessage()" class="bg-indigo-600 text-white px-4 py-2 rounded-r-lg hover:bg-indigo-700 transition">Send</button>
            </div>
        </div>

        <!-- Chat Trigger Button -->
        <button onclick="toggleChat()" class="bg-indigo-600 text-white p-4 rounded-full shadow-xl hover:bg-indigo-700 transition transform hover:scale-110 flex items-center justify-center">
            <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path></svg>
        </button>
    </div>

    <!-- Player Details Modal (Hidden) -->
    <div id="player-modal" class="modal opacity-0 pointer-events-none fixed w-full h-full top-0 left-0 flex items-center justify-center z-50">
        <div class="modal-overlay absolute w-full h-full bg-gray-900 opacity-50" onclick="closePlayerModal()"></div>
        
        <div class="modal-container bg-white w-11/12 md:max-w-md mx-auto rounded shadow-lg z-50 overflow-y-auto transform scale-95 transition-transform duration-300" id="modal-content">
            <div class="modal-content py-4 text-left px-6">
                <!-- Title -->
                <div class="flex justify-between items-center pb-3 border-b">
                    <p class="text-2xl font-bold" id="modal-player-name">Player Name</p>
                    <div class="cursor-pointer z-50" onclick="closePlayerModal()">
                        <svg class="fill-current text-black" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 18">
                            <path d="M14.53 4.53l-1.06-1.06L9 7.94 4.53 3.47 3.47 4.53 7.94 9l-4.47 4.47 1.06 1.06L9 10.06l4.47 4.47 1.06-1.06L10.06 9z"></path>
                        </svg>
                    </div>
                </div>
                <!-- Body -->
                <div class="my-5">
                    <div class="grid grid-cols-2 gap-4 mb-4">
                        <div>
                            <span class="text-xs text-gray-500 uppercase">Position</span>
                            <p class="font-bold" id="modal-player-pos">Forward</p>
                        </div>
                        <div>
                            <span class="text-xs text-gray-500 uppercase">Status</span>
                            <p class="font-bold text-green-600" id="modal-player-status">Ready</p>
                        </div>
                        <div>
                            <span class="text-xs text-gray-500 uppercase">Readiness Score</span>
                            <p class="font-bold" id="modal-player-score">95%</p>
                        </div>
                        <div>
                            <span class="text-xs text-gray-500 uppercase">Fatigue Level</span>
                            <p class="font-bold" id="modal-player-fatigue">3.1%</p>
                        </div>
                    </div>
                    <div class="p-3 bg-gray-100 rounded text-sm text-gray-600 mb-4">
                        <p class="mb-2"><strong>Recent Analysis:</strong> HRV metrics indicate optimal recovery. Sleep quality over last 48h has been high.</p>
                        <p><strong>Recommendation:</strong> Cleared for full training intensity.</p>
                    </div>

                    <!-- AI Coach Section -->
                    <div class="border-t pt-4 grid grid-cols-2 gap-3">
                        <button id="ai-coach-btn" onclick="getAICoachAdvice()" class="w-full py-2 bg-indigo-50 text-indigo-700 font-semibold rounded-lg hover:bg-indigo-100 transition duration-150 flex justify-center items-center text-sm">
                            <span class="mr-2">✨</span> Recovery Coach
                        </button>
                        <!-- NEW Nutrition Feature -->
                        <button id="ai-nutrition-btn" onclick="getNutritionPlan()" class="w-full py-2 bg-green-50 text-green-700 font-semibold rounded-lg hover:bg-green-100 transition duration-150 flex justify-center items-center text-sm">
                            <span class="mr-2">🍎</span> AI Nutritionist
                        </button>
                    </div>
                    <div id="ai-coach-response" class="hidden mt-3 p-3 bg-white border border-indigo-100 rounded-lg text-sm text-gray-700"></div>
                </div>
                <!-- Footer -->
                <div class="flex justify-end pt-2">
                    <button class="px-4 py-2 bg-gray-200 p-3 rounded-lg text-black hover:bg-gray-300 mr-2" onclick="closePlayerModal()">Close</button>
                    <button class="px-4 py-2 bg-indigo-600 p-3 rounded-lg text-white hover:bg-indigo-500">View Full Report</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // --- API CONFIGURATION ---
        const apiKey = ""; // System provided key
        
        async function callGeminiAPI(prompt) {
            const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`;
            const payload = {
                contents: [{ parts: [{ text: prompt }] }]
            };

            for (let i = 0; i < 3; i++) {
                try {
                    const response = await fetch(url, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    
                    const data = await response.json();
                    return data.candidates?.[0]?.content?.parts?.[0]?.text || "No analysis available.";
                } catch (e) {
                    console.error("API Error", e);
                    await new Promise(r => setTimeout(r, Math.pow(2, i) * 1000));
                }
            }
            return "Unable to connect to AI service. Please try again later.";
        }

        // --- AI FEATURES ---

        // 1. Live Vitals Analysis
        async function analyzeLiveVitals() {
            const container = document.getElementById('live-analysis-container');
            const output = document.getElementById('live-analysis-text');
            const title = document.getElementById('analysis-title');
            
            container.classList.remove('hidden');
            title.innerHTML = '<span class="mr-2">⚡</span> Real-time AI Assessment';
            output.innerHTML = '<div class="h-12 w-full loading-shimmer rounded"></div><p class="text-xs text-center text-gray-500 mt-2">Connecting to AI engine...</p>';

            const hr = document.getElementById('live-hr').innerText;
            const fatigue = document.getElementById('live-fatigue').innerText;
            const signal = "98%";

            const prompt = `You are monitoring a high-performance athlete in real-time.
            Current Telemetry:
            - Heart Rate: ${hr} BPM
            - Fatigue Index: ${fatigue}
            - Signal Quality: ${signal}

            Provide a 1-sentence IMMEDIATE assessment of their physical state and a 1-sentence recommendation.
            Tone: Clinical, urgent if needed.`;

            const text = await callGeminiAPI(prompt);
            output.innerHTML = marked.parse(text);
        }

        // 2. Predict Fatigue Trend
        async function predictFatigueTrend() {
            const container = document.getElementById('live-analysis-container');
            const output = document.getElementById('live-analysis-text');
            const title = document.getElementById('analysis-title');
            
            container.classList.remove('hidden');
            title.innerHTML = '<span class="mr-2">🔮</span> Predictive Fatigue Trajectory';
            output.innerHTML = '<div class="h-12 w-full loading-shimmer rounded"></div><p class="text-xs text-center text-gray-500 mt-2">Forecasting metrics...</p>';

            const recentStress = stressChart.data.datasets[0].data.slice(-5).join(', ');
            
            const prompt = `Based on the last 5 minutes of stress markers: [${recentStress}] (Scale 0-100).
            Predict the athlete's fatigue level for the next 30 minutes. 
            Will they bonk (exhaustion) or maintain steady state? 
            Provide a short forecast.`;

            const text = await callGeminiAPI(prompt);
            output.innerHTML = marked.parse(text);
        }

        // 3. Squad Analyst Feature
        async function generateTeamAnalysis() {
            const container = document.getElementById('ai-report-container');
            const output = document.getElementById('ai-report-text');
            
            container.classList.remove('hidden');
            output.innerHTML = '<div class="h-24 w-full loading-shimmer rounded"></div><p class="text-xs text-center text-gray-500 mt-2">Analyzing player metrics...</p>';

            const teamDataForAI = [
                { name: "Player A", position: "Forward", status: "Ready", score: "95%", fatigue: "3.1%" },
                { name: "Player B", position: "Midfielder", status: "Caution", score: "45%", fatigue: "15.5%" },
                { name: "Player C", position: "Defender", status: "Recovery", score: "10%", fatigue: "35.2%" }
            ];

            const prompt = `You are an elite sports performance analyst. 
            Analyze the following player data: ${JSON.stringify(teamDataForAI)}.
            Provide a brief, bulleted strategic report. 
            1. Identify the biggest risk.
            2. Suggest a specific training adjustment for the 'Caution' and 'Recovery' players.
            Keep it professional and concise (max 100 words).`;

            const text = await callGeminiAPI(prompt);
            output.innerHTML = marked.parse(text);
        }

        // 3.5 Match Strategy Generator
        async function generateMatchStrategy() {
            const container = document.getElementById('ai-report-container');
            const output = document.getElementById('ai-report-text');
            
            container.classList.remove('hidden');
            output.innerHTML = '<div class="h-24 w-full loading-shimmer rounded"></div><p class="text-xs text-center text-gray-500 mt-2">Formulating match strategy based on readiness...</p>';

            const teamDataForAI = [
                { position: "Forwards", readiness: "High" },
                { position: "Midfield", readiness: "Medium-Low (Fatigued)" },
                { position: "Defense", readiness: "Critical (Recovery needed)" }
            ];

            const prompt = `You are a Head Coach. Based on this squad status: ${JSON.stringify(teamDataForAI)}.
            Suggest a tactical approach for the upcoming match.
            Should we press high or park the bus? Who needs to be subbed early?
            Keep it to 3 bullet points.`;

            const text = await callGeminiAPI(prompt);
            output.innerHTML = marked.parse(text);
        }

        // 4. AI Coach Feature (Inside Modal)
        async function getAICoachAdvice() {
            const responseDiv = document.getElementById('ai-coach-response');
            const btn = document.getElementById('ai-coach-btn');
            
            responseDiv.classList.remove('hidden');
            responseDiv.innerHTML = '<div class="flex items-center justify-center"><div class="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-600 mr-2"></div> Thinking...</div>';
            btn.disabled = true;

            const name = document.getElementById('modal-player-name').innerText;
            const status = document.getElementById('modal-player-status').innerText;
            const fatigue = document.getElementById('modal-player-fatigue').innerText;

            const prompt = `You are a physiotherapist expert in sports recovery.
            Player: ${name}
            Status: ${status}
            Fatigue Level: ${fatigue}
            
            Give 2 specific, actionable recovery tips for this player right now. Be direct.`;

            const text = await callGeminiAPI(prompt);
            responseDiv.innerHTML = marked.parse(text);
            btn.disabled = false;
        }

        // 4.5 AI Nutritionist
        async function getNutritionPlan() {
            const responseDiv = document.getElementById('ai-coach-response');
            const btn = document.getElementById('ai-nutrition-btn');
            
            responseDiv.classList.remove('hidden');
            responseDiv.innerHTML = '<div class="flex items-center justify-center"><div class="animate-spin rounded-full h-4 w-4 border-b-2 border-green-600 mr-2"></div> Planning Meals...</div>';
            btn.disabled = true;

            const name = document.getElementById('modal-player-name').innerText;
            const fatigue = document.getElementById('modal-player-fatigue').innerText;

            const prompt = `You are a sports nutritionist. 
            Player: ${name} is showing ${fatigue} fatigue.
            Recommend a post-training meal plan for immediate recovery. 
            Include: 
            1. Hydration strategy
            2. Main meal (Carb/Protein balance)
            Keep it short.`;

            const text = await callGeminiAPI(prompt);
            responseDiv.innerHTML = marked.parse(text);
            btn.disabled = false;
        }

        // 5. Investment Memo Generator
        async function generateInvestmentMemo() {
            const container = document.getElementById('investment-memo-container');
            const output = document.getElementById('investment-memo-text');
            
            container.classList.remove('hidden');
            output.innerHTML = '<div class="h-32 w-full loading-shimmer rounded"></div><p class="text-xs text-center text-gray-500 mt-2">Drafting investment thesis from ROI data...</p>';

            const roi = document.getElementById('roiText').innerText;
            const savings = document.getElementById('injurySavings').innerText;
            const uplift = document.getElementById('performanceValue').innerText;
            const size = document.getElementById('teamSizeInput').value;

            const prompt = `Write a professional, persuasive investment memo summary for 'EXRT AI'.
            
            Key Data:
            - Target Client Team Size: ${size} athletes
            - Projected Annual Injury Savings: ${savings}
            - Projected Performance Uplift Value: ${uplift}
            - Calculated ROI Multiple: ${roi}

            Structure:
            1. **Executive Summary**: One powerful sentence.
            2. **The Problem**: High cost of injury/fatigue in pro sports.
            3. **The Solution**: EXRT AI's predictive monitoring.
            4. **Financial Thesis**: Highlight the ROI and savings.
            
            Keep it punchy, business-focused, and formatted in Markdown.`;

            const text = await callGeminiAPI(prompt);
            output.innerHTML = marked.parse(text);
        }

        // NEW: Personal Wellness Advice
        async function getWellnessAdvice() {
            const output = document.getElementById('wellness-response');
            output.classList.remove('hidden');
            output.innerHTML = '<div class="h-12 w-full loading-shimmer rounded"></div><p class="text-xs text-center text-gray-500 mt-2">Analyzing daily biometrics...</p>';

            const prompt = `You are a personal wellness coach for a regular person using a wearable ECG patch.
            Today's Data:
            - HRV: 45ms (Average for this user)
            - Resting Heart Rate: 62 bpm
            - Stress Balance: Medium
            
            Give a friendly, encouraging 2-sentence advice for today. Suggest a simple activity (like walking or breathing).`;

            const text = await callGeminiAPI(prompt);
            output.innerHTML = marked.parse(text);
        }

        // 6. Chat Widget Logic
        function toggleChat() {
            const chat = document.getElementById('chat-window');
            chat.classList.toggle('hidden');
        }

        async function sendChatMessage() {
            const input = document.getElementById('chat-input');
            const msg = input.value.trim();
            if (!msg) return;

            const chatMessages = document.getElementById('chat-messages');
            
            chatMessages.innerHTML += `
                <div class="flex justify-end chat-slide-enter-active">
                    <div class="bg-indigo-600 text-white p-3 rounded-lg rounded-tr-none max-w-[85%]">
                        ${msg}
                    </div>
                </div>`;
            input.value = '';
            chatMessages.scrollTop = chatMessages.scrollHeight;

            const loadingId = 'loading-' + Date.now();
            chatMessages.innerHTML += `
                <div id="${loadingId}" class="flex justify-start chat-slide-enter-active">
                    <div class="bg-gray-200 text-gray-500 p-3 rounded-lg rounded-tl-none max-w-[85%] flex items-center">
                        <div class="animate-bounce mr-1">●</div><div class="animate-bounce delay-100 mr-1">●</div><div class="animate-bounce delay-200">●</div>
                    </div>
                </div>`;
            chatMessages.scrollTop = chatMessages.scrollHeight;

            const hr = document.getElementById('live-hr').innerText;
            const fatigue = document.getElementById('live-fatigue').innerText;
            const roi = document.getElementById('roiText').innerText;
            
            const context = `You are EXRT AI, a helpful dashboard assistant.
            Current Dashboard State:
            - Live Heart Rate: ${hr}
            - Live Fatigue: ${fatigue}
            - Investment ROI: ${roi}
            
            User Question: ${msg}
            
            Answer briefly based on this data.`;

            const responseText = await callGeminiAPI(context);
            
            document.getElementById(loadingId).remove();
            chatMessages.innerHTML += `
                <div class="flex justify-start chat-slide-enter-active">
                    <div class="bg-indigo-100 text-indigo-900 p-3 rounded-lg rounded-tl-none max-w-[85%]">
                        ${marked.parse(responseText)}
                    </div>
                </div>`;
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }


        // --- EXISTING LOGIC ---

        document.getElementById('profile-btn').addEventListener('click', function() {
            const dropdown = document.getElementById('profile-dropdown');
            const isVisible = !dropdown.classList.contains('hidden');
            if (isVisible) {
                dropdown.classList.add('opacity-0', 'scale-95');
                setTimeout(() => dropdown.classList.add('hidden'), 200);
            } else {
                dropdown.classList.remove('hidden');
                setTimeout(() => dropdown.classList.remove('opacity-0', 'scale-95'), 10);
            }
        });

        document.getElementById('mobile-menu-btn').addEventListener('click', function() {
            const menu = document.getElementById('mobile-menu');
            menu.classList.toggle('hidden');
        });

        const ecgData = [];
        for(let i=0; i<50; i++) ecgData.push(0); 

        const ctxEcg = document.getElementById('ecgChart').getContext('2d');
        const ecgChart = new Chart(ctxEcg, {
            type: 'line',
            data: {
                labels: Array.from({length: 50}, (_, i) => i),
                datasets: [{
                    label: 'mV',
                    data: ecgData,
                    borderColor: '#6366F1',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { min: -1.5, max: 2.5, display: false }
                }
            }
        });

        const ctxStress = document.getElementById('stressChart').getContext('2d');
        const stressChart = new Chart(ctxStress, {
            type: 'line',
            data: {
                labels: ['10:00', '10:05', '10:10', '10:15', '10:20', '10:25'],
                datasets: [
                    {
                        label: 'Stress',
                        data: [65, 59, 80, 81, 56, 55],
                        borderColor: '#EF4444',
                        tension: 0.3
                    },
                    {
                        label: 'Meditation Goal',
                        data: [40, 40, 40, 40, 40, 40],
                        borderColor: '#10B981',
                        borderDash: [5, 5],
                        tension: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, position: 'bottom' } },
                scales: { y: { beginAtZero: true, max: 100 } }
            }
        });

        // Initialize Personal Health Chart (Static Data)
        const ctxPersonal = document.getElementById('personalTrendChart').getContext('2d');
        new Chart(ctxPersonal, {
            type: 'bar',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{
                    label: 'Daily HRV Score',
                    data: [42, 45, 40, 48, 52, 45, 47],
                    backgroundColor: '#60A5FA',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true } }
            }
        });

        const heartbeatPattern = [0.1, 0.15, 0.1, 0, -0.2, 1.5, -0.5, 0, 0.2, 0.3, 0.2, 0, 0, 0];
        let patternIndex = 0;

        setInterval(() => {
            let newVal = 0;
            if (patternIndex < heartbeatPattern.length) {
                newVal = heartbeatPattern[patternIndex];
                patternIndex++;
            } else if (Math.random() > 0.85) {
                patternIndex = 0;
            } else {
                newVal = (Math.random() - 0.5) * 0.1;
            }

            ecgChart.data.datasets[0].data.push(newVal);
            ecgChart.data.datasets[0].data.shift(); 
            ecgChart.update();

            if (Math.random() > 0.7) {
                const hr = 70 + Math.floor(Math.random() * 15);
                document.getElementById('live-hr').innerText = hr;
                
                const fatigue = (9 + Math.random()).toFixed(1) + '%';
                document.getElementById('live-fatigue').innerText = fatigue;
            }

        }, 50); 

        const teamInput = document.getElementById('teamSizeInput');
        const salaryInput = document.getElementById('salaryInput');
        
        function calculateROI() {
            const players = parseFloat(teamInput.value) || 0;
            const salary = parseFloat(salaryInput.value) || 0; 
            
            const payroll = players * salary * 1000;
            const injurySavings = payroll * 1.5; 
            const perfValue = payroll * 2.2;
            const totalValue = injurySavings + perfValue;
            
            const cost = players * 20000;
            const roi = (totalValue / (cost || 1)).toFixed(1);

            document.getElementById('injurySavings').innerText = '$' + injurySavings.toLocaleString();
            document.getElementById('performanceValue').innerText = '$' + perfValue.toLocaleString();
            document.getElementById('roiMultiple').innerText = roi + 'x';
            document.getElementById('roiText').innerText = roi + 'x';
        }

        teamInput.addEventListener('input', calculateROI);
        salaryInput.addEventListener('input', calculateROI);

        const modal = document.getElementById('player-modal');
        const modalContent = document.getElementById('modal-content');
        const body = document.querySelector('body');

        function openPlayerModal(name, pos, score, status, fatigue) {
            document.getElementById('modal-player-name').innerText = name;
            document.getElementById('modal-player-pos').innerText = pos;
            document.getElementById('modal-player-score').innerText = score;
            document.getElementById('modal-player-status').innerText = status;
            document.getElementById('modal-player-fatigue').innerText = fatigue;
            
            document.getElementById('ai-coach-response').classList.add('hidden');
            document.getElementById('ai-coach-response').innerHTML = '';
            
            const statusEl = document.getElementById('modal-player-status');
            statusEl.className = 'font-bold';
            if(status === 'Ready') statusEl.classList.add('text-green-600');
            else if(status === 'Caution') statusEl.classList.add('text-yellow-600');
            else statusEl.classList.add('text-red-600');

            modal.classList.remove('opacity-0', 'pointer-events-none');
            modalContent.classList.remove('scale-95');
            modalContent.classList.add('scale-100');
            body.classList.add('modal-active');
        }

        function closePlayerModal() {
            modal.classList.add('opacity-0', 'pointer-events-none');
            modalContent.classList.add('scale-95');
            modalContent.classList.remove('scale-100');
            body.classList.remove('modal-active');
        }
    </script>
</body>
</html>