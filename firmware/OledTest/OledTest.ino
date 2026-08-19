// OLED-only test: scan I2C (SDA=D2/GPIO4, SCL=D1/GPIO5), report over serial, show text.
#include <Wire.h>
#include <U8g2lib.h>
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);

void setup(){
  Serial.begin(115200); delay(600);
  Wire.begin(4, 5);                  // SDA=GPIO4(D2), SCL=GPIO5(D1)
  Serial.println("\n=== OLED I2C test ===");
  int found=0; uint8_t addr=0;
  for(uint8_t a=1;a<127;a++){
    Wire.beginTransmission(a);
    if(Wire.endTransmission()==0){ Serial.printf("  I2C device at 0x%02X\n", a); found++; addr=a; }
  }
  Serial.printf("devices found: %d\n", found);
  if(found){
    u8g2.setI2CAddress(addr<<1);
    u8g2.begin();
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_ncenB12_tr);
    u8g2.drawStr(6, 26, "OLED OK");
    char b[24]; sprintf(b, "addr 0x%02X", addr);
    u8g2.setFont(u8g2_font_ncenB08_tr);
    u8g2.drawStr(6, 50, b);
    u8g2.sendBuffer();
    Serial.println("OLED initialized -> screen should read 'OLED OK'");
  } else {
    Serial.println("NO I2C DEVICE -> check VCC=3V3, GND, SDA=D2, SCL=D1");
  }
}
void loop(){ delay(1000); }
