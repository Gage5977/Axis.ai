import SwiftUI
import Combine

@main
struct AxisTerminal: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 1000, idealWidth: 1200, maxWidth: .infinity,
                       minHeight: 700, idealHeight: 800, maxHeight: .infinity)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.hiddenTitleBar)
    }
}

struct ContentView: View {
    @StateObject private var viewModel = TerminalViewModel()
    @FocusState private var inputFocused: Bool
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            VStack(spacing: 0) {
                // Custom title bar
                HStack {
                    Text("AXIS TERMINAL")
                        .font(.system(size: 16, weight: .ultraLight, design: .monospaced))
                        .tracking(4)
                        .foregroundColor(Color(white: 0.9))
                    
                    Spacer()
                    
                    // Connection status
                    HStack(spacing: 6) {
                        Circle()
                            .fill(viewModel.isConnected ? Color.green : Color.red)
                            .frame(width: 8, height: 8)
                        
                        Text(viewModel.connectionStatus)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(Color(white: 0.6))
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 16)
                .background(Color(white: 0.05))
                
                // Messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(viewModel.messages) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }
                        }
                        .padding(20)
                    }
                    .background(Color.black)
                    .onChange(of: viewModel.messages.count) { _ in
                        withAnimation(.easeOut(duration: 0.3)) {
                            proxy.scrollTo(viewModel.messages.last?.id, anchor: .bottom)
                        }
                    }
                }
                
                // Input area
                HStack(spacing: 16) {
                    TextField("Enter command...", text: $viewModel.input)
                        .textFieldStyle(PlainTextFieldStyle())
                        .font(.system(size: 15, design: .monospaced))
                        .foregroundColor(.white)
                        .padding(14)
                        .background(Color(white: 0.08))
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(inputFocused ? Color.green.opacity(0.5) : Color(white: 0.15), lineWidth: 1)
                        )
                        .focused($inputFocused)
                        .onSubmit {
                            viewModel.sendMessage()
                        }
                    
                    Button(action: viewModel.sendMessage) {
                        Image(systemName: viewModel.isProcessing ? "stop.circle.fill" : "arrow.up.circle.fill")
                            .font(.system(size: 32))
                            .foregroundColor(viewModel.input.isEmpty ? Color(white: 0.3) : Color.green)
                    }
                    .buttonStyle(PlainButtonStyle())
                    .disabled(viewModel.input.isEmpty && !viewModel.isProcessing)
                }
                .padding(20)
                .background(Color(white: 0.03))
            }
        }
        .onAppear {
            inputFocused = true
            viewModel.startConnectionCheck()
        }
    }
}

struct MessageBubble: View {
    let message: Message
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(message.isUser ? "USER" : "AXIS")
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .tracking(1.5)
                    .foregroundColor(message.isUser ? Color.green : Color.cyan)
                
                Spacer()
                
                Text(message.timestamp)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Color(white: 0.4))
            }
            
            Text(message.content)
                .font(.system(size: 14, design: .monospaced))
                .foregroundColor(.white)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(message.isUser ? Color(white: 0.05) : Color(white: 0.02))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(message.isUser ? Color.green.opacity(0.2) : Color.cyan.opacity(0.2), lineWidth: 1)
                )
        )
    }
}

struct Message: Identifiable {
    let id = UUID()
    let content: String
    let isUser: Bool
    let timestamp: String
}

@MainActor
class TerminalViewModel: ObservableObject {
    @Published var messages: [Message] = []
    @Published var input = ""
    @Published var isProcessing = false
    @Published var isConnected = false
    @Published var connectionStatus = "Checking..."
    
    private var cancellables = Set<AnyCancellable>()
    private var availableEndpoints: [(url: String, name: String)] = []
    private var currentEndpoint: String?
    
    init() {
        addMessage("Welcome to AXIS Terminal. Scanning for local AI services...", isUser: false)
    }
    
    func startConnectionCheck() {
        Timer.publish(every: 5, on: .main, in: .common)
            .autoconnect()
            .sink { _ in
                self.checkConnections()
            }
            .store(in: &cancellables)
        
        checkConnections()
    }
    
    private func checkConnections() {
        Task {
            let endpoints = [
                ("http://localhost:3000", "Node API (3000)"),
                ("http://localhost:5001", "Finance ETL (5001)"),
                ("http://localhost:8000", "Python API (8000)"),
                ("http://localhost:8080", "Python API (8080)"),
                ("http://localhost:11434", "Ollama (11434)")
            ]
            
            for (baseUrl, name) in endpoints {
                if await checkEndpoint(baseUrl) {
                    await MainActor.run {
                        self.isConnected = true
                        self.connectionStatus = name
                        self.currentEndpoint = baseUrl
                        if self.messages.count == 1 {
                            self.addMessage("Connected to \(name)", isUser: false)
                        }
                    }
                    return
                }
            }
            
            await MainActor.run {
                self.isConnected = false
                self.connectionStatus = "No services found"
            }
        }
    }
    
    private func checkEndpoint(_ url: String) async -> Bool {
        // Special check for Ollama
        if url.contains("11434") {
            if let tagsUrl = URL(string: url + "/api/tags") {
                do {
                    let (_, response) = try await URLSession.shared.data(from: tagsUrl)
                    if let httpResponse = response as? HTTPURLResponse {
                        return httpResponse.statusCode == 200
                    }
                } catch {}
            }
        }
        
        guard let url = URL(string: url) else { return false }
        
        do {
            let (_, response) = try await URLSession.shared.data(from: url)
            if let httpResponse = response as? HTTPURLResponse {
                return httpResponse.statusCode < 500
            }
        } catch {
            // Try POST endpoints
            let postEndpoints = ["/api/chat", "/chat", "/v1/chat/completions", "/api/generate"]
            for endpoint in postEndpoints {
                if let postUrl = URL(string: url.absoluteString + endpoint) {
                    var request = URLRequest(url: postUrl)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.httpBody = "{}".data(using: .utf8)
                    request.timeoutInterval = 2
                    
                    do {
                        let (_, response) = try await URLSession.shared.data(for: request)
                        if let httpResponse = response as? HTTPURLResponse {
                            return httpResponse.statusCode < 500
                        }
                    } catch {
                        continue
                    }
                }
            }
        }
        
        return false
    }
    
    func sendMessage() {
        if isProcessing {
            // Cancel current request
            isProcessing = false
            return
        }
        
        guard !input.isEmpty, let endpoint = currentEndpoint else { return }
        
        let message = input
        input = ""
        addMessage(message, isUser: true)
        
        isProcessing = true
        
        Task {
            let response = await callAPI(message, endpoint: endpoint)
            await MainActor.run {
                self.addMessage(response, isUser: false)
                self.isProcessing = false
            }
        }
    }
    
    private func callAPI(_ prompt: String, endpoint: String) async -> String {
        // Special handling for Ollama
        if endpoint.contains("11434") {
            let ollamaPayload: [String: Any] = [
                "model": "mistral:latest",
                "prompt": prompt,
                "stream": false
            ]
            
            if let response = await tryAPICall(endpoint: endpoint + "/api/generate", payload: ollamaPayload) {
                return response
            }
        }
        
        // Try different API formats for other services
        let apiPaths = [
            "/api/chat",
            "/chat",
            "/v1/chat/completions",
            "/api/generate",
            "/generate",
            "/"
        ]
        
        let payloads: [[String: Any]] = [
            ["prompt": prompt],
            ["message": prompt],
            ["text": prompt],
            ["query": prompt],
            ["input": prompt],
            ["messages": [["role": "user", "content": prompt]]],
            ["model": "mistral:latest", "prompt": prompt, "stream": false]
        ]
        
        for path in apiPaths {
            for payload in payloads {
                if let response = await tryAPICall(endpoint: endpoint + path, payload: payload) {
                    return response
                }
            }
        }
        
        return "Error: Could not connect to AI service at \(endpoint). Make sure your local AI service is running."
    }
    
    private func tryAPICall(endpoint: String, payload: [String: Any]) async -> String? {
        guard let url = URL(string: endpoint),
              let jsonData = try? JSONSerialization.data(withJSONObject: payload) else {
            return nil
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData
        request.timeoutInterval = 30
        
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return nil
            }
            
            // Extract response from various formats
            if let response = json["response"] as? String { return response }
            if let text = json["text"] as? String { return text }
            if let message = json["message"] as? String { return message }
            if let output = json["output"] as? String { return output }
            if let result = json["result"] as? String { return result }
            if let answer = json["answer"] as? String { return answer }
            if let data = json["data"] as? String { return data }
            
            // OpenAI format
            if let choices = json["choices"] as? [[String: Any]],
               let firstChoice = choices.first,
               let message = firstChoice["message"] as? [String: Any],
               let content = message["content"] as? String {
                return content
            }
            
            // If we got here, we received a response but couldn't parse it
            return "Received response but couldn't parse it: \(json)"
            
        } catch {
            return nil
        }
    }
    
    private func addMessage(_ content: String, isUser: Bool) {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        
        messages.append(Message(
            content: content,
            isUser: isUser,
            timestamp: formatter.string(from: Date())
        ))
    }
}