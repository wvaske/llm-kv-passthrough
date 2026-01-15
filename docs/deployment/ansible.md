# Ansible Deployment

KV-Bench provides Ansible playbooks for automated deployment across multiple servers.

## Prerequisites

- Ansible 2.9+
- SSH access to target servers
- Python 3.10+ on target servers

## Inventory Setup

Create an inventory file `inventory.yml`:

```yaml
all:
  vars:
    kvbench_version: "latest"
    kvbench_model: "llama-3.1-8b"
    kvbench_gpu: "H100_SXM"
    kvbench_storage: "redis"
    redis_url: "redis://redis:6379"

  children:
    redis:
      hosts:
        redis-1:
          ansible_host: 10.0.0.10

    prefill:
      hosts:
        prefill-1:
          ansible_host: 10.0.0.11
          kvbench_instance_id: prefill-1
        prefill-2:
          ansible_host: 10.0.0.12
          kvbench_instance_id: prefill-2

    decode:
      hosts:
        decode-1:
          ansible_host: 10.0.0.21
          kvbench_instance_id: decode-1
        decode-2:
          ansible_host: 10.0.0.22
          kvbench_instance_id: decode-2

    proxy:
      hosts:
        proxy-1:
          ansible_host: 10.0.0.30
          kvbench_instance_id: proxy-1
```

## Playbooks

### Deploy Playbook

`playbooks/deploy.yml`:

```yaml
---
- name: Deploy KV-Bench
  hosts: all
  become: yes
  vars:
    kvbench_user: kvbench
    kvbench_group: kvbench
    kvbench_home: /opt/kvbench
    kvbench_config_dir: /etc/kvbench
    kvbench_log_dir: /var/log/kvbench
    kvbench_data_dir: /var/lib/kvbench

  tasks:
    - name: Create kvbench group
      group:
        name: "{{ kvbench_group }}"
        state: present

    - name: Create kvbench user
      user:
        name: "{{ kvbench_user }}"
        group: "{{ kvbench_group }}"
        home: "{{ kvbench_home }}"
        shell: /bin/bash
        system: yes

    - name: Create directories
      file:
        path: "{{ item }}"
        state: directory
        owner: "{{ kvbench_user }}"
        group: "{{ kvbench_group }}"
        mode: '0755'
      loop:
        - "{{ kvbench_home }}"
        - "{{ kvbench_config_dir }}"
        - "{{ kvbench_log_dir }}"
        - "{{ kvbench_data_dir }}"

    - name: Install Python dependencies
      pip:
        name: kvbench
        version: "{{ kvbench_version }}"
        state: present
      when: kvbench_version != "latest"

    - name: Install latest kvbench
      pip:
        name: kvbench
        state: latest
      when: kvbench_version == "latest"

    - name: Deploy configuration
      template:
        src: config.yaml.j2
        dest: "{{ kvbench_config_dir }}/config.yaml"
        owner: "{{ kvbench_user }}"
        group: "{{ kvbench_group }}"
        mode: '0640'
      notify: Restart kvbench

    - name: Deploy systemd service
      template:
        src: kvbench.service.j2
        dest: /etc/systemd/system/kvbench.service
        mode: '0644'
      notify:
        - Reload systemd
        - Restart kvbench

    - name: Enable and start kvbench
      systemd:
        name: kvbench
        enabled: yes
        state: started

  handlers:
    - name: Reload systemd
      systemd:
        daemon_reload: yes

    - name: Restart kvbench
      systemd:
        name: kvbench
        state: restarted
```

### Configure Playbook

`playbooks/configure.yml`:

```yaml
---
- name: Configure KV-Bench
  hosts: all
  become: yes

  tasks:
    - name: Update configuration
      template:
        src: config.yaml.j2
        dest: /etc/kvbench/config.yaml
        owner: kvbench
        group: kvbench
        mode: '0640'
      notify: Restart kvbench

  handlers:
    - name: Restart kvbench
      systemd:
        name: kvbench
        state: restarted
```

### Benchmark Playbook

`playbooks/benchmark.yml`:

```yaml
---
- name: Run KV-Bench Benchmarks
  hosts: proxy
  become: yes
  vars:
    benchmark_output_dir: /var/lib/kvbench/benchmarks
    benchmark_duration: 300
    benchmark_concurrency: 10

  tasks:
    - name: Create benchmark output directory
      file:
        path: "{{ benchmark_output_dir }}"
        state: directory
        owner: kvbench
        group: kvbench
        mode: '0755'

    - name: Run GenAI-Perf benchmark
      shell: |
        genai-perf \
          --endpoint http://localhost:8000/v1/chat/completions \
          --model {{ kvbench_model }} \
          --concurrency {{ benchmark_concurrency }} \
          --duration {{ benchmark_duration }} \
          --output {{ benchmark_output_dir }}/genai-perf-{{ ansible_date_time.iso8601_basic_short }}.json
      args:
        chdir: "{{ benchmark_output_dir }}"
      register: benchmark_result
      ignore_errors: yes

    - name: Collect metrics
      uri:
        url: http://localhost:8000/metrics
        method: GET
        return_content: yes
      register: metrics

    - name: Save metrics
      copy:
        content: "{{ metrics.json | to_nice_json }}"
        dest: "{{ benchmark_output_dir }}/metrics-{{ ansible_date_time.iso8601_basic_short }}.json"

    - name: Fetch benchmark results
      fetch:
        src: "{{ benchmark_output_dir }}/{{ item }}"
        dest: ./benchmark-results/
        flat: yes
      with_fileglob:
        - "{{ benchmark_output_dir }}/*.json"
```

### Teardown Playbook

`playbooks/teardown.yml`:

```yaml
---
- name: Teardown KV-Bench
  hosts: all
  become: yes

  tasks:
    - name: Stop kvbench service
      systemd:
        name: kvbench
        state: stopped
        enabled: no
      ignore_errors: yes

    - name: Remove systemd service
      file:
        path: /etc/systemd/system/kvbench.service
        state: absent
      notify: Reload systemd

    - name: Remove kvbench package
      pip:
        name: kvbench
        state: absent

    - name: Remove directories
      file:
        path: "{{ item }}"
        state: absent
      loop:
        - /opt/kvbench
        - /etc/kvbench
        - /var/log/kvbench
        - /var/lib/kvbench

    - name: Remove kvbench user
      user:
        name: kvbench
        state: absent
        remove: yes

    - name: Remove kvbench group
      group:
        name: kvbench
        state: absent

  handlers:
    - name: Reload systemd
      systemd:
        daemon_reload: yes
```

## Templates

### Configuration Template

`templates/config.yaml.j2`:

```yaml
instance_id: {{ kvbench_instance_id | default(inventory_hostname) }}

server:
  host: 0.0.0.0
  port: 8000
  server_type: {{ kvbench_server_type | default('combined') }}
  model_profile: {{ kvbench_model }}
  workers: {{ kvbench_workers | default(1) }}
  log_level: {{ kvbench_log_level | default('info') }}

gpu:
  gpu_profile: {{ kvbench_gpu }}
  efficiency_factor: {{ kvbench_efficiency | default(0.7) }}
  tp_size: {{ kvbench_tp_size | default(1) }}

storage:
  backend_type: {{ kvbench_storage }}
{% if kvbench_storage == 'redis' %}
  redis_url: {{ redis_url }}
  redis_cluster: {{ redis_cluster | default(false) | lower }}
{% elif kvbench_storage == 'nfs' or kvbench_storage == 'weka' %}
  filesystem_path: {{ kvbench_filesystem_path }}
{% endif %}

{% if kvbench_server_type == 'proxy' %}
distributed:
  prefill_endpoints:
{% for host in groups['prefill'] %}
    - http://{{ hostvars[host].ansible_host }}:8000
{% endfor %}
  decode_endpoints:
{% for host in groups['decode'] %}
    - http://{{ hostvars[host].ansible_host }}:8000
{% endfor %}
  health_check_interval: 10.0
{% endif %}

metrics:
  enabled: true
  prometheus_port: 9090
```

### Systemd Service Template

`templates/kvbench.service.j2`:

```ini
[Unit]
Description=KV-Bench Server
After=network.target
{% if kvbench_storage == 'redis' %}
Wants=redis.service
{% endif %}

[Service]
Type=simple
User=kvbench
Group=kvbench
WorkingDirectory=/opt/kvbench
ExecStart=/usr/local/bin/kvbench serve --config /etc/kvbench/config.yaml
Restart=always
RestartSec=10
StandardOutput=append:/var/log/kvbench/kvbench.log
StandardError=append:/var/log/kvbench/kvbench.log

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/kvbench /var/log/kvbench

[Install]
WantedBy=multi-user.target
```

## Usage

### Deploy All

```bash
ansible-playbook -i inventory.yml playbooks/deploy.yml
```

### Deploy Specific Group

```bash
# Deploy only prefill servers
ansible-playbook -i inventory.yml playbooks/deploy.yml -l prefill

# Deploy only decode servers
ansible-playbook -i inventory.yml playbooks/deploy.yml -l decode
```

### Run Benchmarks

```bash
ansible-playbook -i inventory.yml playbooks/benchmark.yml
```

### Update Configuration

```bash
ansible-playbook -i inventory.yml playbooks/configure.yml
```

### Teardown

```bash
ansible-playbook -i inventory.yml playbooks/teardown.yml
```

## Best Practices

1. **Use Ansible Vault** for sensitive data (Redis passwords, etc.)
2. **Tag tasks** for selective execution
3. **Test in staging** before production deployment
4. **Monitor rollouts** with health checks
5. **Keep backups** of configuration before changes
