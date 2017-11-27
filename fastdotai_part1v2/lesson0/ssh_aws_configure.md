1. Create key pair 
2. Open  https://console.aws.amazon.com/ec2/
3. Go to EC2, click on  Key Pairs and Create Key Pair
4. Move your *pem file to a save place such as directory .shh
    ```sh
    chmod 400 my-key-pair.pem
    ```
2. Launch an instance with `Deep Learning AMI`
3. Look for Public DNS (IPv4)
4. It looks like `ec2-54-191-181-145.us-west-2.compute.amazonaws.com`
5. Use you `*.pem` file to ssh to your instance
    ```sh
    ssh -i /Users/yinterian/.ssh/aws-key.pem ubuntu@ec2-54-186-214-175.us-west-2.compute.amazonaws.com
    ```
    Or
    ```sh
    ssh -i .ssh/aws-key.pem ubuntu@54.186.214.175
    ```

### Tunneling:
```sh
ssh -i .ssh/aws-key.pem ubuntu@54.186.214.175 -L 8888:127.0.0.1:8888
localhost:8888
```sh
