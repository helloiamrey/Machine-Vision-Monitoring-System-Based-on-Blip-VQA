import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

my_sender = '2996963411@qq.com'  # 填写发信人的邮箱账号
my_pass = 'hgdgvwuhkzimdfad'  # 发件人邮箱授权码
my_user = '114514.iamrey@gmail.com'  # 收件人邮箱账号


def send_mail():
    ret = True
    try:
        msg = MIMEText('监控系统检测到潜在的紧急情况（如人员摔倒），请立即查看！\n\n此邮件由自动监控系统发送。', 'plain', 'utf-8')  # 填写邮件内容
        msg['From'] = formataddr(["BOT", my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        msg['To'] = formataddr(["默认紧急联系人", my_user])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msg['Subject'] = "紧急情况警报：监控系统检测到异常"  # 邮件的主题，也可以说是标题

        server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱授权码
        server.sendmail(my_sender, [my_user, ], msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()  # 关闭连接
    except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
        ret = False
    if ret:
        print("邮件发送成功")
    else:
        print("邮件发送失败")
    return ret


#ret = mail()
#if ret:
#    print("邮件发送成功")
#else:
#    print("邮件发送失败")