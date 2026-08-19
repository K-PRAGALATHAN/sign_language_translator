// Identify the ESP32-CAM sensor: scan bus + read chip-ID registers (AI-Thinker).
#include <Wire.h>
#define PWDN_GPIO_NUM  32
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27

void startXclk(int pin, uint32_t freq){ ledcSetup(0, freq, 1); ledcAttachPin(pin,0); ledcWrite(0,1); }

uint8_t rd16(uint8_t addr, uint16_t reg){         // 16-bit reg addr (OV3660/OV5640)
  Wire.beginTransmission(addr); Wire.write(reg>>8); Wire.write(reg&0xFF);
  if(Wire.endTransmission(false)!=0) return 0xE1;
  Wire.requestFrom(addr,(uint8_t)1); return Wire.available()?Wire.read():0xE2;
}
uint8_t rd8(uint8_t addr, uint8_t reg){            // 8-bit reg addr (OV2640)
  Wire.beginTransmission(addr); Wire.write(reg);
  if(Wire.endTransmission(false)!=0) return 0xE1;
  Wire.requestFrom(addr,(uint8_t)1); return Wire.available()?Wire.read():0xE2;
}

void setup(){
  Serial.begin(115200); delay(800);
  Serial.println("\n=== ESP32-CAM sensor ID ===");
  pinMode(PWDN_GPIO_NUM,OUTPUT); digitalWrite(PWDN_GPIO_NUM,LOW); delay(10);
  Wire.begin(SIOD_GPIO_NUM,SIOC_GPIO_NUM,100000);
  startXclk(XCLK_GPIO_NUM,20000000); delay(80);

  int found=0;
  for(uint8_t a=1;a<127;a++){ Wire.beginTransmission(a); if(Wire.endTransmission()==0){ Serial.printf("ACK 0x%02X\n",a); found++; } }
  Serial.printf("devices: %d\n", found);

  // OV3660/OV5640 use 16-bit regs 0x300A/0x300B for chip id
  uint8_t h16=rd16(0x3C,0x300A), l16=rd16(0x3C,0x300B);
  Serial.printf("16-bit ID @0x3C: 0x%02X%02X  (OV3660=0x3660, OV5640=0x5640)\n", h16,l16);
  // OV2640 style 8-bit regs at 0x30
  uint8_t h8=rd8(0x30,0x0A), l8=rd8(0x30,0x0B);
  Serial.printf("8-bit ID @0x30:  0x%02X%02X  (OV2640=0x26xx)\n", h8,l8);
  Serial.println("=== done ===");
}
void loop(){ delay(2000); }
