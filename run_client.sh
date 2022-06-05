# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조
# sh run.sh => 실행
#!/bin/bash

# client 수 설정
for i in `seq 0 2`; do
    echo "Starting client $i"
    python /Users/yangsemo/VScode/Flower_Health/Health_Flower_Client/client.py --partition=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait