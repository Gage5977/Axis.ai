import SwiftUI
import Combine

@main
struct AITerminalPro: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 900, idealWidth: 1200, maxWidth: .infinity,
                       minHeight: 700, idealHeight: 800, maxHeight: .infinity)
        }
        .windowStyle(.hiddenTitleBar)
    }
}

struct ContentView: View {
    @StateObject private var viewModel = TerminalViewModel()
    @FocusState private var inputFocused: Bool
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack(spacing: 16) {
                Text("AXIS TERMINAL")
                    .font(.system(size: 18, weight: .ultraLight, design: .monospaced))
                    .tracking(3)
                    .foregroundColor(Color(hex: "E0E0E0"))
                
                Spacer()
                
                // Connection status
                Circle()
                    .fill(viewModel.isConnected ? Color(hex: "00FF88") : Color(hex: "FF4444"))
                    .frame(width: 8, height: 8)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 16)
            .background(Color(hex: "0A0A0A"))
            
            // Messages area
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(viewModel.messages) { message in
                            MessageView(message: message)
                                .id(message.id)
                        }
                        
                        if viewModel.isStreaming {
                            StreamingIndicator()
                        }
                    }
                    .padding(24)
                }
                .background(Color(hex: "050505"))
                .onChange(of: viewModel.messages.count) { _ in
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(viewModel.messages.last?.id, anchor: .bottom)
                    }
                }
            }
            
            // Input area
            VStack(spacing: 0) {
                Divider()
                    .background(Color(hex: "1A1A1A"))
                
                HStack(alignment: .center, spacing: 16) {
                    TextField("Enter command...", text: $viewModel.inputText)
                        .textFieldStyle(PlainTextFieldStyle())
                        .font(.system(size: 14, weight: .light, design: .monospaced))
                        .foregroundColor(Color(hex: "E0E0E0"))
                        .padding(12)
                        .background(Color(hex: "0F0F0F"))
                        .cornerRadius(6)
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(Color(hex: "1A1A1A"), lineWidth: 1)
                        )
                        .focused($inputFocused)
                        .onSubmit {
                            viewModel.sendMessage()
                        }
                    
                    Button(action: viewModel.sendMessage) {
                        Image(systemName: viewModel.isProcessing ? "stop.circle.fill" : "arrow.up.circle.fill")
                            .font(.system(size: 28))
                            .foregroundColor(viewModel.isProcessing ? Color(hex: "FF4444") : 
                                           (viewModel.inputText.isEmpty ? Color(hex: "404040") : Color(hex: "00FF88")))
                    }
                    .buttonStyle(PlainButtonStyle())
                    .disabled(!viewModel.isProcessing && viewModel.inputText.isEmpty)
                }
                .padding(20)
                .background(Color(hex: "080808"))
            }
            
            // Status bar
            HStack(spacing: 16) {
                Text(viewModel.statusText)
                    .font(.system(size: 11, weight: .light, design: .monospaced))
                    .foregroundColor(Color(hex: "808080"))
                
                Spacer()
                
                if viewModel.tokensPerSecond > 0 {
                    Text("\(Int(viewModel.tokensPerSecond)) tokens/s")
                        .font(.system(size: 11, weight: .light, design: .monospaced))
                        .foregroundColor(Color(hex: "00FF88"))
                }
                
                Text(viewModel.latency)
                    .font(.system(size: 11, weight: .light, design: .monospaced))
                    .foregroundColor(Color(hex: "808080"))
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 8)
            .background(Color(hex: "0A0A0A"))
        }
        .background(Color(hex: "000000"))
        .onAppear {
            inputFocused = true
            viewModel.checkConnection()
        }
    }
}

struct MessageView: View {
    let message: Message
    @State private var isCopied = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(message.role == .user ? "USER" : "AXIS")
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .tracking(1)
                    .foregroundColor(message.role == .user ? Color(hex: "00FF88") : Color(hex: "00AAFF"))
                
                if let timestamp = message.timestamp {
                    Text(timestamp)
                        .font(.system(size: 10, weight: .light, design: .monospaced))
                        .foregroundColor(Color(hex: "606060"))
                }
                
                Spacer()
                
                if message.role == .assistant {
                    Button(action: { copyToClipboard(message.content) }) {
                        Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 12))
                            .foregroundColor(Color(hex: "606060"))
                    }
                    .buttonStyle(PlainButtonStyle())
                }
            }
            
            Text(message.content)
                .font(.system(size: 14, weight: .light, design: .monospaced))
                .foregroundColor(Color(hex: "E0E0E0"))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 16)
        .background(message.role == .user ? Color(hex: "0A0F0A") : Color(hex: "0A0A0F"))
        .cornerRadius(8)
    }
    
    private func copyToClipboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        
        withAnimation(.easeInOut(duration: 0.2)) {
            isCopied = true
        }
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            withAnimation(.easeInOut(duration: 0.2)) {
                isCopied = false
            }
        }
    }
}

struct StreamingIndicator: View {
    @State private var opacity = 0.3
    
    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { index in
                Circle()
                    .fill(Color(hex: "00AAFF"))
                    .frame(width: 6, height: 6)
                    .opacity(opacity)
                    .animation(
                        .easeInOut(duration: 0.6)
                        .repeatForever()
                        .delay(Double(index) * 0.2),
                        value: opacity
                    )
            }
        }
        .padding(.leading, 16)
        .padding(.vertical, 8)
        .onAppear {
            opacity = 1.0
        }
    }
}

struct Message: Identifiable {
    let id = UUID()
    let role: Role
    let content: String
    let timestamp: String?
    
    enum Role {
        case user, assistant, system
    }
}

@MainActor
class TerminalViewModel: ObservableObject {
    @Published var messages: [Message] = []
    @Published var inputText = ""
    @Published var isProcessing = false
    @Published var isStreaming = false
    @Published var statusText = "Initializing..."
    @Published var currentModel = "mistral:latest"
    @Published var isConnected = false
    @Published var tokensPerSecond: Double = 0
    @Published var latency = "0ms"
    
    private var streamTask: Task<Void, Never>?
    private let dateFormatter: DateFormatter
    
    init() {
        dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss"
        
        messages.append(Message(
            role: .system,
            content: "AXIS Terminal initialized. Ready for input.",
            timestamp: dateFormatter.string(from: Date())
        ))
    }
    
    func checkConnection() {
        Task {
            do {
                let task = Process()
                task.executableURL = URL(fileURLWithPath: "/usr/bin/which")
                task.arguments = ["ollama"]
                
                let pipe = Pipe()
                task.standardOutput = pipe
                
                try task.run()
                task.waitUntilExit()
                
                await MainActor.run {
                    isConnected = task.terminationStatus == 0
                    statusText = isConnected ? "Connected to Ollama" : "Ollama not found"
                }
                
            } catch {
                await MainActor.run {
                    isConnected = false
                    statusText = "Connection check failed"
                }
            }
        }
    }
    
    func sendMessage() {
        if isProcessing {
            // Cancel current stream
            streamTask?.cancel()
            streamTask = nil
            isProcessing = false
            isStreaming = false
            statusText = "Cancelled"
            return
        }
        
        guard !inputText.isEmpty else { return }
        
        let userMessage = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        inputText = ""
        
        messages.append(Message(
            role: .user,
            content: userMessage,
            timestamp: dateFormatter.string(from: Date())
        ))
        
        isProcessing = true
        statusText = "Processing..."
        
        streamTask = Task {
            await streamResponse(for: userMessage)
        }
    }
    
    private func streamResponse(for prompt: String) async {
        let startTime = Date()
        var responseText = ""
        var tokenCount = 0
        
        isStreaming = true
        
        // Create response message
        let responseMessage = Message(
            role: .assistant,
            content: "",
            timestamp: dateFormatter.string(from: Date())
        )
        messages.append(responseMessage)
        
        do {
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            task.arguments = ["ollama", "run", currentModel, prompt]
            
            let pipe = Pipe()
            task.standardOutput = pipe
            task.standardError = pipe
            
            try task.run()
            
            let handle = pipe.fileHandleForReading
            
            // Stream output
            while task.isRunning {
                let data = handle.availableData
                if !data.isEmpty, let chunk = String(data: data, encoding: .utf8) {
                    responseText += chunk
                    tokenCount += chunk.split(separator: " ").count
                    
                    // Update message
                    if let index = messages.firstIndex(where: { $0.id == responseMessage.id }) {
                        messages[index] = Message(
                            role: .assistant,
                            content: responseText,
                            timestamp: responseMessage.timestamp
                        )
                    }
                    
                    // Update metrics
                    let elapsed = Date().timeIntervalSince(startTime)
                    if elapsed > 0 {
                        tokensPerSecond = Double(tokenCount) / elapsed
                    }
                }
                
                try await Task.sleep(nanoseconds: 10_000_000) // 10ms
            }
            
            // Get remaining data
            let finalData = handle.readDataToEndOfFile()
            if let finalChunk = String(data: finalData, encoding: .utf8) {
                responseText += finalChunk
            }
            
            // Final update
            if let index = messages.firstIndex(where: { $0.id == responseMessage.id }) {
                messages[index] = Message(
                    role: .assistant,
                    content: responseText.isEmpty ? "No response received" : responseText,
                    timestamp: responseMessage.timestamp
                )
            }
            
        } catch {
            if let index = messages.firstIndex(where: { $0.id == responseMessage.id }) {
                messages[index] = Message(
                    role: .assistant,
                    content: "Error: \(error.localizedDescription)",
                    timestamp: responseMessage.timestamp
                )
            }
        }
        
        // Update final metrics
        let totalTime = Date().timeIntervalSince(startTime)
        latency = String(format: "%.0fms", totalTime * 1000)
        
        isProcessing = false
        isStreaming = false
        statusText = "Ready"
        streamTask = nil
    }
}

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}