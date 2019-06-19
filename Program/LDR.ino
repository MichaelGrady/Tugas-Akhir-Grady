//KEY_ESC  0xB1  177
void setup()

{
Serial.begin(9600);
}

void loop()   
{
int AnalogValue;
while (!Serial.available())
{
AnalogValue = analogRead(A0);

Serial.println(AnalogValue);
delay(100);

}
}
