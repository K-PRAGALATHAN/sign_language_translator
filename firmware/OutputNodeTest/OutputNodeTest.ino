// Output-node self-test for NodeMCU ESP8266:
//  - SSD1306 OLED over I2C (SCL=D1/GPIO5, SDA=D2/GPIO4)  -> shows text
//  - MAX98357A speaker over I2S (DIN=GPIO3, BCLK=GPIO15/D8, LRC=GPIO2/D4) -> beeps
// If the OLED shows text AND you hear beeps, both peripherals work.
#include <Arduino.h>
#include <U8g2lib.h>
#include <core_esp8266_i2s.h>
#include <math.h>

// Software I2C so we can pin SCL/SDA explicitly
U8G2_SSD1306_128X64_NONAME_F_SW_I2C u8g2(U8G2_R0, /*SCL=*/5, /*SDA=*/4, U8X8_PIN_NONE);

void show(const char* a, const char* b) {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_ncenB10_tr);
  u8g2.drawStr(0, 22, a);
  u8g2.drawStr(0, 48, b);
  u8g2.sendBuffer();
}

void tone_i2s(float freq, int ms) {
  const int rate = 16000;
  long n = (long)rate * ms / 1000;
  for (long i = 0; i < n; i++) {
    int16_t s = (int16_t)(sinf(2.0f * PI * freq * i / rate) * 10000);
    i2s_write_lr(s, s);  // blocks when the DMA buffer is full
  }
  for (int i = 0; i < 256; i++) i2s_write_lr(0, 0);  // flush silence
}

void setup() {
  Serial.begin(115200);
  Serial.println("\nOutput-node test starting");
  u8g2.begin();
  show("SLT TEST", "OLED OK");
  delay(1200);
  i2s_begin();
  i2s_set_rate(16000);
  Serial.println("I2S ready");
}

void loop() {
  show("Speaker", "BEEP 1kHz");
  Serial.println("beep");
  tone_i2s(1000, 350);
  show("SLT TEST", "waiting...");
  delay(900);
}
