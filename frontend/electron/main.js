const { app, BrowserWindow, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let viteProcess;

const isDev = !app.isPackaged;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 850,
    minWidth: 1024,
    minHeight: 768,
    title: 'Gastro AI - Teşhis ve Analiz Portalı',
    icon: path.join(__dirname, '..', 'public', 'favicon.svg'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
    backgroundColor: '#0b0f19',
    show: false,
    autoHideMenuBar: true,
  });

  // Remove default menu bar
  Menu.setApplicationMenu(null);

  if (isDev) {
    // Development: load from Vite dev server
    const loadURL = () => {
      mainWindow.loadURL('http://localhost:5173').catch(() => {
        // Vite not ready yet, retry in 500ms
        setTimeout(loadURL, 500);
      });
    };
    loadURL();
  } else {
    // Production: load built files
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function startViteDev() {
  if (!isDev) return Promise.resolve();

  return new Promise((resolve) => {
    viteProcess = spawn('npx', ['vite', '--port', '5173'], {
      cwd: path.join(__dirname, '..'),
      shell: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    viteProcess.stdout.on('data', (data) => {
      const output = data.toString();
      process.stdout.write(output);
      if (output.includes('Local:') || output.includes('ready')) {
        resolve();
      }
    });

    viteProcess.stderr.on('data', (data) => {
      process.stderr.write(data.toString());
    });

    // Fallback resolve after 8 seconds
    setTimeout(resolve, 8000);
  });
}

app.whenReady().then(async () => {
  await startViteDev();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (viteProcess) {
    viteProcess.kill();
  }
  app.quit();
});

app.on('before-quit', () => {
  if (viteProcess) {
    viteProcess.kill();
  }
});
