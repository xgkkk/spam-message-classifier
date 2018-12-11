<script type="text/javascript">
     var myChart = echarts.init(document.getElementById('main'));

    //初始化数据
    var barData = [{{spam_p}}, {{notspam_p}}];

    var option = {
        title: {
            text: '是否是垃圾邮件的概率分布',
            left: 'center'
        },

        xAxis: {
            type: 'value',
            axisLine: {
                show: false
            },
            axisTick: {
                show: false
            },
            nameTextStyle: {
                fontSize: 30
            }
        },
        yAxis: {
            type: 'category',
            data: ["否", "是"],
            splitLine: {show: false},
            axisLine: {
                show: false
            },
            axisTick: {
                show: false
            },
            offset: 10,
        },
        series: [
            {
                name: '数量',
                type: 'bar',
                data: barData,
                barWidth: 20,
                barGap: 10,
                smooth: true,
                label: {
                    normal: {
                        show: true,
                        position: 'right',
                        <!--offset: [5, -2],-->
                        textStyle: {
                            color: 'red',
                            fontSize: 16
                        }
                    }
                },
            }
        ]
    };
    myChart.setOption(option);
</script>
