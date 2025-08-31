const google_myanmar_tools = require('myanmar-tools');
const converter = new google_myanmar_tools.ZawgyiConverter();
const unicodeText = converter.zawgyiToUnicode(process.argv[2]);
console.log(unicodeText);
