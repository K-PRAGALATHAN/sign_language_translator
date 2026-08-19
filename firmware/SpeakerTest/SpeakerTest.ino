// Speaker-only test: MAX98357A over I2S. Plays an ascending 3-tone chime on repeat.
// I2S pins (fixed on ESP8266): DIN=GPIO3(RX), BCLK=GPIO15(D8), LRC=GPIO2(D4).
#include <U8g2lib.h>
#include <core_esp8266_i2s.h>
#include <math.h>

U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);

void tone_i2s(float freq, int ms){
  const int rate = 16000;
  long n = (long)rate * ms / 1000;
  for(long i=0;i<n;i++){
    int16_t s = (int16_t)(sinf(2.0f*PI*freq*i/rate) * 14000);   // loud
    i2s_write_lr(s, s);
  }
  for(int i=0;i<400;i++) i2s_write_lr(0,0);                     // trailing silence
}

void show(const char* a, const char* b){
  u8g2.clearBuffer(); u8g2.setFont(u8g2_font_ncenB10_tr);
  u8g2.drawStr(4,24,a); u8g2.drawStr(4,50,b); u8g2.sendBuffer();
}

void setup(){
  Serial.begin(115200); delay(500);
  u8g2.setI2CAddress(0x3C<<1); u8g2.begin();
  show("SPEAKER","test");
  i2s_begin(); i2s_set_rate(16000);
  Serial.println("\nSpeaker test: ascending chime on loop");
}

void loop(){
  Serial.println("chime");
  show("SPEAKER","BEEP 1");  tone_i2s(500, 250);
  show("SPEAKER","BEEP 2");  tone_i2s(1000,250);
  show("SPEAKER","BEEP 3");  tone_i2s(1500,250);
  show("SPEAKER","listen...");
  delay(900);
}
