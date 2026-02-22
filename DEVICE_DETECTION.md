# Automatic Device Detection

## Overview

The Digital Wellbeing Predictor now **automatically detects** whether each connected device is mobile or desktop, eliminating the need for manual layout toggling.

## How It Works

### 1. Detection Process

When a user opens the application:

1. **JavaScript Detection**: On first page load, JavaScript checks both:
   - **User-Agent string**: `navigator.userAgent` for mobile keywords (iPhone, Android, etc.)
   - **Viewport width**: `window.innerWidth < 768` for narrow screens
2. **Query Parameter Method**: Detection result is saved as URL query parameter `?mobile_detected=1` (mobile) or `?mobile_detected=0` (desktop)
3. **Page Reload**: JavaScript triggers one automatic reload with the detection parameter
4. **Layout Assignment**: Python reads the query param and sets `st.session_state.mobile_layout`:
   - `True` → Mobile-optimized UI (stacked columns, tab-based inputs)
   - `False` → Desktop-optimized UI (side-by-side columns, inline controls)

**Why query parameters?**  
Streamlit's `components.html()` doesn't support direct return values to Python. Using query parameters allows JavaScript to communicate the detection result reliably across the page reload.

### 2. Multi-Device Support

Each browser session gets **independent detection**:

- **Laptop** connecting → Detects desktop User-Agent → Desktop layout
- **iPhone** connecting → Detects iOS User-Agent → Mobile layout
- **Android tablet** connecting → Detects Android User-Agent → Mobile layout

Sessions are isolated, so different devices get appropriate layouts simultaneously.

### 3. UI Differences

**Mobile Layout (iOS/Android):**
- Stacked vertical columns for narrow screens
- Tab-based input vs results (Stress Predictor page)
- Lighter chart density
- Touch-friendly controls

**Desktop Layout (Laptop/PC):**
- Side-by-side columns for wide screens
- Inline input and results
- Higher chart density
- Mouse-optimized controls

## Code Changes

### Previous Behavior (Manual Toggle)
```python
# Sidebar had manual toggle button
st.session_state.mobile_layout = st.toggle(
    "Mobile layout (iOS / Android)",
    value=st.session_state.mobile_layout,
    help="Enable a phone-friendly stacked UI..."
)
```

### New Behavior (Auto-Detection)
```python
# Automatic detection via JavaScript viewport/User-Agent detection
def _auto_detect_device_layout() -> None:
    if "mobile_layout" not in st.session_state:
        # Check query params set by JavaScript
        params = st.query_params
        if "mobile_detected" in params:
            st.session_state.mobile_layout = params["mobile_detected"][0] == "1"
            return
        
        # Inject detection JavaScript that sets query param and reloads
        components.html("""
            <script>
            const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) 
                          || window.innerWidth < 768;
            const params = new URLSearchParams(window.location.search);
            params.set('mobile_detected', isMobile ? '1' : '0');
            window.location.href = url.toString();
            </script>
        """, height=0)

# Called once in main() before UI render
_auto_detect_device_layout()
```

## Testing

### Desktop Browsers
User-Agent examples that trigger **desktop layout**:
```
Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0
Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Safari/537.36
Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Firefox/121.0
```

### Mobile Browsers
User-Agent examples that trigger **mobile layout**:
```
Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) Safari/604.1
Mozilla/5.0 (Linux; Android 13; Pixel 7) Chrome/120.0.0.0 Mobile Safari/537.36
Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) Safari/604.1
```

### Browser DevTools Testing
1. Open browser DevTools (F12)
2. Toggle Device Emulation (Ctrl+Shift+M)
3. Select mobile device (iPhone 14, Pixel 5, etc.)
4. Reload page → Page reloads once with `?mobile_detected=1` → Mobile layout applied
5. Switch to "Responsive" desktop mode
6. Clear query param or reload → Page detects desktop → Desktop layout applied

**Note:** You'll see the URL briefly change to include `?mobile_detected=0` or `?mobile_detected=1` - this is the detection mechanism communicating the result.

## Sidebar Display

The sidebar shows the **detected device type** (informational only):

```
Device: Loading...       (before detection)
Device: Detecting...     (detection in progress)
Device: 📱 Mobile        (detected on iPhone)
Device: 💻 Desktop       (detected on laptop)
```

This replaces the previous manual toggle button.

**First-load behavior:** On the very first page load, you may briefly see "Detecting..." before the page automatically reloads once with the detection result. After that, the layout persists for the entire session.

## Fallback Behavior

If User-Agent detection fails (e.g., privacy browser blocking navigator.userAgent):
- Defaults to **desktop layout** (`mobile_layout = False`)
- User still gets full functionality, just with desktop UI

## Benefits

✅ **No Manual Setup**: Users don't need to toggle layout  
✅ **Per-Session Detection**: Each device gets appropriate UI automatically  
✅ **Better UX**: Mobile users get optimized touch interface  
✅ **Deployment Ready**: Works for multi-device simultaneous access  

## Implementation Files

- **Main detection logic**: `app.py` lines 1003-1083
  - `_detect_mobile_from_user_agent()`: Pattern matching function
  - `_auto_detect_device_layout()`: JavaScript probe and detection
  - `render_sidebar()`: Displays detected device type
  - `main()`: Calls detection before UI render

- **Layout consumers**:
  - `_render_device_cards()`: Uses `mobile_layout` for column stacking
  - `page_stress_predictor()`: Uses `mobile_layout` for tab-based input
  - Various chart rendering functions check `st.session_state.mobile_layout`

## Migration Notes

**What Changed:**
- ❌ Removed: Manual toggle button in sidebar
- ✅ Added: Automatic User-Agent detection
- ✅ Added: Device type display (informational)
- ✅ Preserved: All existing mobile/desktop layout logic

**Backward Compatibility:**
- Session state key `mobile_layout` remains the same
- All existing layout-checking code works unchanged
- Only initialization method changed (manual → automatic)

---

**Author**: 214129X — Malalpola MLHR  
**Course**: CM4371 Machine Learning and Pattern Recognition  
**Feature**: Automatic OS/Device Detection for Responsive UI
