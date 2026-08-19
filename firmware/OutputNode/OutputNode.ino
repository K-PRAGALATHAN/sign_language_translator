// SLT Output Node (NodeMCU ESP8266): receives a translated sentence + its TTS audio
// from the laptop over WiFi, shows the text on the OLED, and plays the audio on the
// MAX98357A speaker.
//
// Protocol (raw TCP on port 9001):
//   line 1: sentence text        (newline-terminated, UTF-8/ASCII)
//   line 2: sample rate in Hz     (ASCII integer, newline-terminated)
//   then  : raw 16-bit signed little-endian MONO PCM, until the client closes.
//
// OLED: SSD1306 I2C @0x3C  (SDA=D2/GPIO4, SCL=D1/GPIO5)
// I2S : DIN=GPIO3(RX), BCLK=D8/GPIO15, LRC=D4/GPIO2
#include <ESP8266WiFi.h>
#include <U8g2lib.h>
#include <core_esp8266_i2s.h>

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
const uint16_t TCP_PORT = 9001;

U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
WiFiServer server(TCP_PORT);

void oledStatus(const char* l1, const char* l2) {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x12_tr);
  u8g2.drawStr(0, 12, l1);
  if (l2) u8g2.drawStr(0, 28, l2);
  u8g2.sendBuffer();
}

// Word-wrap a sentence to the 128px display (up to 5 lines).
void oledSentence(const String& text) {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x12_tr);
  const int maxW = 128, lineH = 13, maxLines = 5;
  int y = 12, lines = 0;
  String line = "", word = "";
  String t = text + " ";
  for (unsigned i = 0; i < t.length() && lines < maxLines; i++) {
    char c = t[i];
    if (c == ' ') {
      String trial = line.length() ? line + " " + word : word;
      if (u8g2.getStrWidth(trial.c_str()) > maxW && line.length()) {
        u8g2.drawStr(0, y, line.c_str()); y += lineH; lines++;
        line = word;
      } else {
        line = trial;
      }
      word = "";
    } else {
      word += c;
    }
  }
  if (lines < maxLines && line.length()) u8g2.drawStr(0, y, line.c_str());
  u8g2.sendBuffer();
}

String readLine(WiFiClient& c, uint32_t timeoutMs = 4000) {
  String s; uint32_t t = millis();
  while (millis() - t < timeoutMs) {
    if (c.available()) {
      char ch = c.read();
      if (ch == '\n') break;
      if (ch != '\r') s += ch;
      t = millis();
    } else {
      yield();
    }
  }
  return s;
}

void streamAudio(WiFiClient& client) {
  uint8_t buf[512];
  bool half = false; uint8_t halfByte = 0;
  uint32_t lastData = millis();
  while (client.connected() || client.available()) {
    int n = client.available();
    if (n <= 0) {
      if (millis() - lastData > 3000) break;   // sender stalled/finished
      yield();
      continue;
    }
    if (n > (int)sizeof(buf)) n = sizeof(buf);
    int r = client.read(buf, n);
    lastData = millis();
    int j = 0;
    if (half) {                                 // finish a sample split across reads
      int16_t s = (int16_t)((uint16_t)halfByte | ((uint16_t)buf[0] << 8));
      i2s_write_lr(s, s);
      j = 1; half = false;
    }
    for (; j + 1 < r; j += 2) {
      int16_t s = (int16_t)((uint16_t)buf[j] | ((uint16_t)buf[j + 1] << 8));
      i2s_write_lr(s, s);
    }
    if (r - j == 1) { halfByte = buf[j]; half = true; }
  }
  for (int i = 0; i < 600; i++) i2s_write_lr(0, 0);   // flush the DMA tail
}

void setup() {
  Serial.begin(115200);
  u8g2.setI2CAddress(0x3C << 1);
  u8g2.begin();
  oledStatus("WiFi...", WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("\nConnecting to "); Serial.println(WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  String ip = WiFi.localIP().toString();
  Serial.print("\nOUTPUT_NODE_IP: "); Serial.println(ip);

  server.begin();
  i2s_begin();
  i2s_set_rate(16000);

  oledStatus("Ready", ip.c_str());
}

void loop() {
  WiFiClient client = server.available();
  if (!client) { delay(5); return; }

  String text = readLine(client);
  uint32_t rate = readLine(client).toInt();
  if (rate < 4000 || rate > 48000) rate = 16000;
  Serial.printf("recv: \"%s\"  rate=%u\n", text.c_str(), rate);

  if (text.length()) oledSentence(text);
  i2s_set_rate(rate);
  streamAudio(client);
  client.stop();
  Serial.println("done");
}
