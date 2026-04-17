#!/usr/bin/env python3
"""
聊天相关路由
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
from backend.models import ChatRequest, ChatResponse
from backend.services.chat_service import ChatService
from backend.logging_config import get_logger
import json
from pathlib import Path
import uuid
import os
import sys
import PyPDF2
import requests
from bs4 import BeautifulSoup

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

router = APIRouter(prefix="/chat", tags=["聊天"])
logger = get_logger(__name__)
