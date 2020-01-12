import shutil
import os
import imaplib
import email

directory = r'C:\Users\heman\Desktop\Projects\Recipe project\data\test'

def delete_files():
	try:
		if os.path.exists(directory):
			shutil.rmtree(directory,ignore_errors=True) #it deletes even if the dir is in read only mode
	except OSError:
		print('Error: Deleting directory. '+directory)

def create_dir():
	try:
		if not os.path.exists(directory):
			os.mkdir(directory,0o755) #0o755 gives write access
	except OSError:
		print('Error: Creating directory. '+directory)

def access_mail():
	mail = imaplib.IMAP4_SSL("imap.mail.yahoo.com",993)
	mail.login(username, generate password) #you have to get the password from account settings
	mail.select('INBOX')
	type, data = mail.search(None, 'ALL')
	mail_ids = data[0]
	id_list = mail_ids.split()
	for num in data[0].split():
		typ, data = mail.fetch(num,'(RFC822)')
		raw_email = data[0][1]
	# converts byte literal to string removing b''
	raw_email_string = raw_email.decode('utf-8')
	email_message = email.message_from_string(raw_email_string)
	# downloading attachments
	for part in email_message.walk(): 
		if part.get_content_maintype() == 'multipart':
			continue
		if part.get('Content-Disposition') is None:
			continue
		fileName = part.get_filename()
		if bool(fileName):
			filePath = directory+'/'+fileName
			if not os.path.isfile(filePath) :
				fp = open(filePath, 'wb')
				fp.write(part.get_payload(decode=True))
				fp.close()


if __name__ == '__main__':
	delete_files()
	create_dir()
	access_mail()