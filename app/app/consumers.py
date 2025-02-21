#from channels.generic.websocket import AsyncWebsocketConsumer
#import json

# class TrainingConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         await self.channel_layer.group_add("training_updates", self.channel_name)
#         await self.accept()

#     async def disconnect(self, close_code):
#         await self.channel_layer.group_discard("training_updates", self.channel_name)
 
#     async def training_start(self, event):
#         await self.send(json.dumps({"type": "training.start"}))

#     async def training_progress(self, event):
#         await self.send(json.dumps({"type": "training.progress", "progress": event["progress"]}))

#     async def training_end(self, event):
#         await self.send(json.dumps({"type": "training.end"}))


# class ChatConsumer(AsyncWebsocketConsumer):
#     def connect(self):
#         self.accept()

#         self.send(text_data=json.dumps({
#             'type':'connectionn_established',
#             'message':'You are now connected!'
#         }))


import json 
#from channels.generic.websocket import WebsocketConsumer

# class ChatConsumer(WebsocketConsumer):
#     def connect(self):
#         self.accept()

#         self.send(text_data=json.dumps({
#             'type':'connection_established',
#             'message':'You are now connected!'
#         }))