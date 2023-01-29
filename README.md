1A) Aim : Add two Numbers
1. Open UiPath Studio and click on Blank to start a fresh project. Give it a meaningful name. Like AddTwoNumber
2. Open Main.xaml from Project tab. On the Designer panel, double click a Sequence activity from the Activities panel.
3. Select Variable tab from bottom of page. Create variable x and y and select variable data type as int32 and save.
4. Select and drag input dialog from activities. Fill all data labels. Add entered value x and y variable respectively. As below.
5. select and drag Messagebox from activity menu. Write x+y and save.

1B) Aim: Create project to show ODD and even using flowchart.
1. Open UiPath Studio and click on Blank to start a fresh project. Give it a meaningful name. Like ShowOddEventFlowchart
2. Open Main.xaml from Project tab. On the Designer panel, double click a flowchart activity from the Activities panel.
Add next activity flow decision. Set condition like (x mod 2=0) in properties selection.
3. Add two more message box and connect with true and false flow. And give even number is its true. Give odd number if its false.

1C) Create an UiPath Robot which can empty recycle bin in Gmail solely on basis of recording. Aim: Use Web Recorder to empty spam in gmail.
1. Open UiPath Studio and click on Blank to start a fresh project. Give it a meaningful name.
2. Open gmail in browser and select spam from right side menu.
3. Select App/Web recording from UI path Design menu and select the steps from gmail -> spam -> delete all spam.

==============================================================================================================================================

2A) Automate UiPath Number Calculation (Subtraction, Multiplication, Division of numbers).
1. Add sequence on page then Select 2 input dialog for two number from activity panel and create variable x and y with int32 datatype.
2. Select 4 message box from activity for add, sub, mul, div.

2B) Create an automation UiPath project using different types of variables (number, datetime, Boolean, generic, array, data table)
1. Open UiPath Studio and click on Blank to start a fresh project. Give it a meaningful name.
2. Create data table from Build Data table.
3. Select message box from activity for int32,Boolean, String datatype.
4. Select for each activity for Int32[] array and message box to show value.
5. Select for each for data table activity for datatable variable which is type of DataTable and message box to show value.

==============================================================================================================================================

3A) Create an automation UiPath Project using looping statements. 
1. Open UiPath Studio and click on Blank to start a fresh project. Give it a meaningful name. Like Pratical3B.
2. Open Main.xaml from Project tab. Select sequence, then create variable CoutryList with variable type Array of [T] type String.
3. Add foreach from activity window. Add variable as below. Then add messagebox in body to show value.

==============================================================================================================================================

4A) Automate any process using basic recording
1: Open UI path and create new project with appropriate name and choose language type VB.
2: Open notepad
3: Click on App/Web recorder
4: Click on notepad. Select white area and type anything. And save process.

4B) Automate any process using desktop recording
1: Open UI path and create new project with appropriate name and choose language type VB.
2: Open notepad
3: Click on App/Web recorder
4: Click on notepad. Select white area and type anything. And save process.
5: Close the notepad.
O/P : Also automatically save with desktoprecording1.txt

4C) Automate any process using web recording
1: Open UI path and create new project with appropriate name and choose language type VB.
2: Click and drag open browser from the activity panel.
3: Enter Url of the form under double quotation.

==============================================================================================================================================

5) Aim: Consider an array of names. We have to find out how many of these start with the letter “a” Create an automation where the number of names starting with “a”
 is counted and the result is displayed.

1: Open UI path and create new project with appropriate name and choose language type VB.
2: Add sequence in project from activity panel. Create variable “names”. Variable type Array of [T] String.
3: Default values {"vijay","ajay","rahul","ashish","arman","akash","vipul"}.
4: Add “for each” from activities panel. In = names
5: Add if inside “for each”. Add condition like currentItem.ToString.StartsWith("a").
6: Assign and increment variable counter.

==============================================================================================================================================

6A) Create an application automating the read, write and append operation on excel file.
1 : Create an Excel file with Name, Address, Phone, Comment and save the file.
2: open UiPath and create a new blank project with appropriate name.
3: Click on Sequence in the activity panel. Now Click on “Open browser” activity and paste the url of the form.
4: Click on “Excel Application scope” and browse the excel file.
5: Take the Read Range activity from the activity panel.
6: click on “for each row in datatable” from activity panel in=data1. And create variable data1.
7: Take “sequence” activity from activity panel. After that take “type into” activity.
Indicate on screen
Click on Name.
Repeat the same for Address, mob, and comment. Type “CurrentRow.item("Name").ToString” in the
Type into field.
8: Take click event. Click on Indicate on Screen event and indicate it on submit.
9: Add a delay, Duration HH:MM:SS=00:00:02.
10:Take Click event, indicate on screen = back tab and Add Delay again.
Save the project and execute.

==============================================================================================================================================

7a) Implement the attach window activity.
1) Open UiPath Studio and click on Blank to start a fresh project. Give it a meaningful name. Like Pratical7A.
2) Open Main.xaml from Project tab. On the Designer panel, double click a flowchart activity from the Activities panel.
3) Create a sequence and set it as Start node.
4) Drag and drop attach window activity and indicate an untitled word document.
5) In the do section add type into activity to insert some text and two click activities to close the word by clicking on don’t save option.

7b) Find different controls using UiPath.
1) Create a sequence and set it as start node.
2) Drag and drop open browser and specify the url.
3) In the do section use the activity Element Exists and indicate it at Google Search button and store the Boolean value in Exists attribute.
4) Use the message box to display the Boolean value.
5) Drag and drop the Find Element attribute and indicate it at Google image and the Element in FoundElement attribute.
6) To get the name of UI element use get attribute and specify the Element and Attribute value.
7) Use the message box to display the name.

7c) Demonstrate the following activities in UiPath:
i. Mouse (click, double click and hover)
ii. Type into
iii. Type Secure text

1) Create a flowchart activity.
2) Drag and drag a sequence activity from activities panel.
3) Connect the start node to this sequence
4) Drag and drop the double click activity and indicate the folder icon to open it.
5) Drag and drop another double click activity and indicate the Login page to open it in browser.
6) Drag and drop type into activity and indicate the text area of username in the page and pass the value in double quotes. 
7) Drag and drop the type secure text activity.
8) Create a variable pwd with data type as SecureString and provide the value using Net package and convert it into SecureString and pass this value in SecureText of the input of
type secure text.
9) Drag and drop the hover button to point at Submit button.
10) Drag and drop another click button and indicate it on the submit button.

==============================================================================================================================================

8A) Demonstrate the following events in UiPath:
i. Element triggering event
1) Open UiPath Studio and click on Blank to start a fresh project. Give it a meaningful name. Like Pratical8A.
2) Open Main.xaml from Project tab. On the Designer panel, double click a flowchart activity from the Activities panel.
3) Create a sequence and set it as Start node.
4) Drag and drop a trigger scope activity and in triggers add click trigger – indicate an untitled notepad and specify the mouse button.
5) In actions section’s sequence add a type into activity - indicate an untitled notepad and add some text.

ii) Image triggering event
1) Create another sequence and set this as Start node.
2) Drag and drop a trigger scope activity and in triggers section add click image trigger – indicate a region of image and specify mouse button.
3) In action’s sequence section add a message box and enter some text to display.

iii) System Triggering Event.
1) Create another sequence and set this as Start node.
2) Drag and drop a trigger scope activity and in triggers section add system trigger – check both
keyboard and mouse from its properties panel.
3) In action’s sequence section add a Write Line activity and enter some text to display.

8B) Automate the following screen scraping methods using UiPath
i. Full Text
ii. Native
iii. OCR
1) Open UiPath Studio and click on Blank to start a fresh project. Give it a meaningful name. Like Pratical8B.
2) Open Main.xaml from Project tab. On the Designer panel, double click a flowchart activity from the Activities panel.
3) Click on Screen Scraping option from design tab and specify the region from which we need to extract the information.
4) Specify scraping methods as full text and click on finish.
5) Repeat step 3 and 4 by changing methods as Native and OCR.

8C) Install and automate any process using UiPath with the following plug-ins:
i) Java Plugin
1) Create Java Swing Application through NetBeans.
public class NewJFrame extends javax.swing.JFrame {
 public NewJFrame() {
 initComponents(); }
@SuppressWarnings("unchecked")
private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {
 jLabel3.setText("Submitted successfully!"); }
public static void main(String args[]) {
 java.awt.EventQueue.invokeLater(new Runnable() {
 public void run() {
 new NewJFrame().setVisible(true); } }); }
 private javax.swing.JButton jButton1;
 private javax.swing.JLabel jLabel1;
 private javax.swing.JTextField jTextField1; }
2) Create a Sequence and set it as start node.
3) Drag type into activities and indicate on text areas as shown below.
4) Then click on submit button.
5) If the java plugin is installed properly then in UI explorer the cls will display value as SunAwtFrame.

ii) Mail plugin
1) Create a sequence and set it as Start node.
2) Drag and drop a get password activity type the password in its properties panel and store the output in result by using a string variable.
3) Drag and drop a Send SMTP Mail Message activity and enter to, subject and body values.
4) Provide port as 587 and server as smtp.gmail.com in the host section of properties panel of send smtp mail message.
5) Provide Email(sender) and its corresponding password (by using the variable stored in get password activity) in the Logon section of properties panel of send smtp mail message.
6) Add a message box to inform the user that the mail is sent.

iii) PDF Plugin
1) Create a sequence and set it as Start node.
2) Drag and drop a Read PDF Text activity and specify the path of the pdf which is needed to be read and store its output in text attribute.
3) Use a message box to display the output

iv) Web Integration
1) Create a sequence and set it as Start node.
2) Drag and drop a HTTP Request and add the values as provided below.
3) Use Write Line to display the output.

v) Excel Plugin
1) Create a sequence and set it as start node.
2) Drag and drop a excel application scope and specify the path of the excel file which is to be used.
3) Add a Read Cell activity specify the Sheet1 name and cell as A1 and store the output in result attribute.
4) Use the message box to display the output

vi) Word Plugin
1) Create a sequence and set it as start node.
2) Drag and drop a word application scope and specify the path of the word file which is to be used.
3) Add a Append Text activity specify the text that needs to appended to the document.

vii) Credential Management.
1) Create Generic credentials from the control center.
2) Create a sequence and set it as start node.
3) Drag and drop the Get secure credentials and fill the values as shown below.
4) Use the assign to convert the Secure String pass variable into String datatype by using: new System.Net.NetworkCredential("",pass).Password
5) Use type into and click activities to fill username and password and proceed further to sign in into google account.

==============================================================================================================================================

9A) Automate the process of send mail event (on any email).
1) Open UiPath Studio and click on Blank to start a fresh project. Give it a meaningful name. Like Pratical9.
2) Open Main.xaml from Project tab. On the Designer panel, double click a flowchart activity from the Activities panel.
3) Create a sequence and set it as Start node.
4) Drag and drop a get password activity type the password in its properties panel and store the output in result by using a string variable.
5) Drag and drop a Send SMTP Mail Message activity and enter to, subject and body values.
6) Provide port as 587 and server as smtp.gmail.com in the host section of properties panel of send smtp mail message.
7) Provide Email(sender) and its corresponding password (by using the variable stored in get password activity) in the Logon section of properties panel of send smtp mail message.
8) Add a message box to inform the user that the mail is sent

b) Automate the process of launching an assistant bot on a keyboard event.
1) Create a Sequence and set it as start node.
2) Drag and drop trigger scope activity and in triggers section add hotkey trigger check the Alt key and from dropdown select key as enter.
3) In the action’s sequence use attach window and indicate a untitled notepad and in its do section add a type into activity and specify the text that needs to be written.
4) Add two click activities to close the notepad by clicking on don’t save option.

c) Demonstrate the Exception handing in UiPath.
1) Create a Sequence and set it as start node.
2) Drag and drop a Try Catch activity.
3) In Try section create an array of integers by declaring the arr variable with datatype as
System.Int32[]
4) Use a message box to display the integer at 5th index this will raise an exception as
IndexOutOfRangeException.
5) In the Catches section click on Add new catch and choose IndexOutOfRangeException.
6) Use a message box with text Array has elements till 4th index.
7) If we try to assess value after 4th element the exception section’s message box will be
displayed otherwise the value of the array with specified range will be displayed.

d) Demonstrate the use of config files in UiPath.
1) Create a sequence and set it as start node.
2) Drag and drop an excel application scope activity and specify the path.
3) In the do section drag and drop a read range activity specify the Sheet and range.
4) Use the assign activity to store the values by using the syntax as: dt_Test.Rows(<rownumber>).Item("<Column Name>").
5) Use the message boxes to display the values

==============================================================================================================================================

10A) Automate the process of logging and taking screenshots in UiPath.
1) Drag and drop a Try Catch activity.
2) In Try section create an array of integers by declaring the arr variable with datatype as System.Int32[]
3) Use a message box to display the integer at 5th index this will raise an exception as IndexOutOfRangeException.
4) In the Catches section click on Add new catch and choose IndexOutOfRangeException.
5) Use a message box with text Array has elements till 4th index.
6) If we try to assess value after 4th element the exception section’s message box will be displayed otherwise the value of the array with specified range will be displayed.
7) Use three log message activity to display messages in output panel.
8) Add take screenshot activity save it result in a variable and use save image activity to save the image at the required destination.

b. Automate any process using State Machine in UiPath.
1) Drag and drop state machine activities, state and final state activities as shown below:
2) Create the following variables:
3) Configure each state and trigger activity as shown below

b. Demonstrate the use of publish utility.
1) Add a message box and display hello message.
2) Click on publish present in design tab.
3) Follow the below steps
