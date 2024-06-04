package Tools;

public class string {
    public static String intToAlphabet(int i) {
        if (i > 0 && i < 27) {
            return String.valueOf((char)(i + 'A' - 1)); //integer to char manipulation and then to String
        } else {
            return null;
        }
    }
}
